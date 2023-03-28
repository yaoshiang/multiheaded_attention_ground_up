"""Ground up implementation of multi-headed attention."""

# Quiet down TF's logging.
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import itertools
from pprint import pprint

import numpy as np
import tensorflow as tf

# Force lazy malloc of GPU memory.
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
  tf.config.experimental.set_memory_growth(d, True)

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
RANDGEN = tf.random.Generator.from_non_deterministic_state()

NCLASSES = 10
BATCHSIZE = 512


def get_ds(batch_size, train_per, valid_per, shape=(224, 224, 3)):
  """Creates the Imagenette dataset."""
  datasets = tfds.load(
      "imagenette",
      split=[f"train[:{train_per}%]", f"validation[:{valid_per}%]"],
      shuffle_files=True,
      with_info=False,
      as_supervised=True,
      download=True,
  )

  dataset_train, dataset_valid = datasets

  dataset_train = dataset_train.map(
      lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label),
      num_parallel_calls=AUTOTUNE,
  )
  dataset_valid = dataset_valid.map(
      lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label),
      num_parallel_calls=AUTOTUNE,
  )
  # Alexnet:
  # Preprocessing: resize 1:1 so that the short edge is 256, then center crop to 256x256.
  # Augmentation: Take random 224x224 crops for train; for valid/test, take the four corners + center at 224x224.

  assert shape[0] == 224

  def _alexnet_resize_one_hot(img, label):
    orig_hwc = tf.shape(img)
    if orig_hwc[0] > orig_hwc[1]:
      img = tf.transpose(img, [1, 0, 2])

    hwc = tf.cast(tf.shape(img), tf.float32)
    new_h = 256
    new_w = tf.cast(hwc[1] / hwc[0] * 256.0, tf.int32)

    img = tf.image.resize(img, (new_h, new_w))

    x = new_w - 256
    img = img[:, x // 2:x // 2 + 256, :]

    if orig_hwc[0] > orig_hwc[1]:
      img = tf.transpose(img, [1, 0, 2])

    return img, tf.one_hot(label, 10, dtype=tf.float32)

  dataset_train = dataset_train.map(
      lambda img, label: _alexnet_resize_one_hot(img, label),
      num_parallel_calls=AUTOTUNE,
  )
  dataset_valid = dataset_valid.map(
      lambda img, label: _alexnet_resize_one_hot(img, label),
      num_parallel_calls=AUTOTUNE,
  )

  dataset_train = dataset_train.cache(
      f"/tmp/cached_imagenette_train_{train_per}_alexnet")

  dataset_train = dataset_train.shuffle(1000)

  def _alexnet_crop_train(img):
    y = RANDGEN.uniform(shape=(), minval=0, maxval=32, dtype=tf.int32)
    x = RANDGEN.uniform(shape=(), minval=0, maxval=32, dtype=tf.int32)
    return img[y:y + 224, x:x + 224, :]

  # def _alexnet_crop_valid(ds: tf.data.Dataset) -> tf.data.Dataset:
  #   ds = ds.map(lambda img, label: (
  #       tf.stack((
  #           (img[0:224, 0:224, :]),  # NW
  #           (img[0:224, 32:256, :]),  # NE
  #           (img[16:240, 16:240, :]),  # Center
  #           (img[32:256, 0:224, :]),  # SW
  #           (img[32:256, 32:256, :]),  # SE
  #       )),
  #       tf.tile([label], [5]),
  #   ), AUTOTUNE)
  #   ds = ds.unbatch()
  #   return ds

  def _alexnet_crop_valid_center(ds: tf.data.Dataset) -> tf.data.Dataset:
    ds = ds.map(lambda img, label: (img[16:240, 16:240, :], label), AUTOTUNE)
    return ds

  dataset_train = dataset_train.map(
      lambda img, label: (_alexnet_crop_train(img), label),
      num_parallel_calls=AUTOTUNE,
  )
  dataset_valid = _alexnet_crop_valid_center(dataset_valid)

  dataset_valid = dataset_valid.cache(
      f"/tmp/cached_imagenette_valid_{valid_per}_alexnet")

  dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
  dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True)

  dataset_train = dataset_train.prefetch(AUTOTUNE)
  dataset_valid = dataset_valid.prefetch(AUTOTUNE)

  nvalid = len(dataset_valid)
  dataset_test = dataset_valid.skip(nvalid // 2)
  dataset_valid = dataset_valid.take(nvalid // 2)

  return dataset_train, dataset_valid, dataset_test


class ImageTokenizer(tf.keras.layers.Layer):

  def __init__(self, patch_size, token_length):
    super().__init__()
    self.P = patch_size
    self.token_length = token_length
    self.dense = tf.keras.layers.Dense(token_length)

  def build(self, input_shape):
    self.B = input_shape[0]
    self.H = input_shape[1]
    self.W = input_shape[2]

  @staticmethod
  def chip_image(image, B, H, W, P):
    """Chops an image into chips.

    Args:
      image: A tensor of shape [batch_size, height, width, 3].
      batch_size: An integer.
      height: An integer.
      width: An integer.
      patch_size: An integer.

    Returns:
      A tensor of shape [batch_size, height // patch_size, width // patch_size,
      patch_size, patch_size, 3].
    """
    tf.debugging.assert_equal(H, W)

    # Chunk the rows to B, H, W // P, P, 3.
    image = tf.reshape(image, (-1, H, W // P, P, 3))

    # Push the rows to the end of the axis. B, W//P, P, H, 3.
    image = tf.transpose(image, (0, 2, 3, 1, 4))

    # Chunk the columns to B, W//P, P, H//P, P, 3.
    image = tf.reshape(image, (-1, W // P, P, H // P, P, 3))

    # push the chips to the end of the axis. B, W//P, H//P, P, P, 3.
    image = tf.transpose(image, (0, 1, 3, 4, 2, 5))

    return image

  def call(self, image):
    chips = self.chip_image(image, self.B, self.H, self.W, self.P)

    num_chips = chips.shape[1] * chips.shape[2]

    # Turn the chips into tokens. B, H//P x W // P, P x P x 3.
    image = tf.reshape(image, (-1, num_chips, self.P * self.P * 3))

    tf.debugging.assert_equal(tf.shape(image)[2], 1024 * 3)

    return self.dense(image)


class MultiHeadedAttention(tf.keras.layers.Layer):

  def __init__(self, vector_length, num_heads):
    super().__init__()
    self.K = tf.keras.layers.Dense(1)
    self.Q = tf.keras.layers.Dense(1)
    self.V = tf.keras.layers.Dense(vector_length)

    self.vector_length = vector_length
    self.num_heads = num_heads

    self.layer_norm1 = tf.keras.layers.LayerNormalization()
    self.layer_norm2 = tf.keras.layers.LayerNormalization()

    self.activation = tf.keras.layers.ReLU()

  def build(self, input_shape):
    self.B = input_shape[0]
    self.T = input_shape[1]

  @staticmethod
  def _call(token, K, Q, V, LN1, LN2, A):
    # Keras dense layers add an extra dimension at the end.
    key = K(token)[:, :, 0]
    query = Q(token)[:, :, 0]
    value = V(token)

    kq = tf.einsum('BT, BX -> BTX', query, key)
    index = tf.nn.softmax(kq, axis=2)

    tf.debugging.assert_near(tf.reduce_sum(index[0, 0]), 1.0)

    z = tf.einsum('BTX, BXE -> BTE', index, value)
    z = LN1(z)
    z = LN2(V(z) + z)
    z = A(z)

    return z

  def call(self, token):
    # Token is shape BTL.
    z = self._call(token, self.K, self.Q, self.V, self.layer_norm1,
                   self.layer_norm2, self.activation)

    return z


def get_model(model_name) -> tf.keras.models.Model:
  x = input = tf.keras.Input(shape=(224, 224, 3))
  x = ImageTokenizer(32, 256)(x)
  for layer in range(4):
    x = MultiHeadedAttention(256, 8)(x)
  x = tf.keras.layers.Flatten()(x)
  y = tf.keras.layers.Dense(10)(x)
  model = tf.keras.Model(inputs=input, outputs=y)
  model.build(input_shape=(None, 224, 224, 3))
  model.summary()

  o = tfa.optimizers.LAMB(learning_rate=0.001)

  model.compile(
      optimizer=o,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
  )
  model.optimizer.lr = 0.001
  return model


def main():
  train, val, test = get_ds(batch_size=64,
                            train_per=90,
                            valid_per=10,
                            shape=(224, 224, 3))
  model = get_model("test_model")

  model.fit(train, epochs=10, validation_data=val)


if __name__ == '__main__':
  main()