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

  # dataset_valid = dataset_valid.cache(
  #     f"/tmp/cached_imagenette_valid_{valid_per}_alexnet")

  dataset_train = dataset_train.batch(batch_size, drop_remainder=False)
  dataset_valid = dataset_valid.batch(batch_size, drop_remainder=False)

  dataset_train = dataset_train.prefetch(AUTOTUNE)
  dataset_valid = dataset_valid.prefetch(AUTOTUNE)

  nvalid = len(dataset_valid)
  assert nvalid >= 2
  dataset_test = dataset_valid.skip(nvalid // 2)
  dataset_valid = dataset_valid.take(nvalid // 2)

  return dataset_train, dataset_valid, dataset_test


class ImageTokenizer(tf.keras.layers.Layer):

  def __init__(self, patch_size, token_length):
    super().__init__()
    self.P = patch_size
    self.token_length = token_length

  def build(self, input_shape):
    self.B = input_shape[0]
    self.H = input_shape[1]
    self.W = input_shape[2]

    self.embedding = tf.keras.layers.Dense(self.token_length)
    self.positional = tf.keras.layers.Dense(self.token_length)

  @staticmethod
  def chip_image(image, B, H, W, P):
    """Chops an image into chips.

    TODO: Test a looping implementation. Although this algo is vectorized,
    it relies on the very expensive transpose operation. It may be
    faster to loop in code, and allow TF to unroll the loop.
    
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

    looping = True

    if looping:
      image = tf.reshape(image, (-1, H, W * 3))

      chips = []
      for row in range(H // P):
        y_beg = row * P
        y_end = (row + 1) * P
        for col in range(W // P):
          x_beg = col * P * 3
          x_end = (col + 1) * P * 3
          chip = image[:, y_beg:y_end, x_beg:x_end]
          chips.append(tf.reshape(chip, (-1, P * P * 3)))

      chips = tf.stack(chips, axis=1)
      return chips
    else:
      # Chunk the rows to B, H, W // P, P, 3.
      image = tf.reshape(image, (-1, H, W // P, P, 3))

      # Push the rows to the end of the axis. B, W//P, P, H, 3.
      image = tf.transpose(image, (0, 2, 3, 1, 4))

      # Chunk the columns to B, W//P, P, H//P, P, 3.
      image = tf.reshape(image, (-1, W // P, P, H // P, P, 3))

      # push the chips to the end of the axis. B, W//P, H//P, P, P, 3.
      image = tf.transpose(image, (0, 3, 1, 4, 2, 5))

      # Flatten the chips. B, H//P x W // P, P x P x 3.
      image = tf.reshape(image, (-1, H // P * W // P, P * P * 3))

      return image

  def call(self, image):
    chips = self.chip_image(image, self.B, self.H, self.W, self.P)

    num_chips = chips.shape[1]

    # BCL (Batch, Chips, Embedding Length)
    embedding = self.embedding(chips)

    positions = tf.range(num_chips, dtype=tf.float32)  # C
    positions = positions / num_chips  # C
    positions = positions[..., tf.newaxis]  # C1
    positional_embedding = self.positional(positions)  # CL

    # BCL + CL -> BCL
    return embedding + positional_embedding


class MultiHeadedAttention(tf.keras.layers.Layer):

  def __init__(self, embedding_length, num_heads):
    super().__init__()

    self.embedding_length = embedding_length
    self.num_heads = num_heads

    self.layer_norm1 = tf.keras.layers.LayerNormalization()
    self.layer_norm2 = tf.keras.layers.LayerNormalization()

    self.activation = tfa.layers.GELU()

  def build(self, input_shape):
    if len(input_shape) == 3:
      # BTE
      self.token_input = True
    elif len(input_shape) == 4:
      # BHTE
      self.token_input = False
    else:
      assert False

    self.B = input_shape[0]
    self.T = input_shape[1]

    H = self.num_heads
    E = self.embedding_length
    T = self.T

    self.key_kernel = self.add_weight(
        name='key_kernel',
        shape=(1, H, E),
        initializer=tf.keras.initializers.glorot_normal(),
        trainable=True)
    self.query_kernel = self.add_weight(
        name='query_kernel',
        shape=(1, H, E),
        initializer=tf.keras.initializers.glorot_normal(),
        trainable=True)
    self.value_kernel = self.add_weight(
        name='value_kernel',
        shape=(1, H, E, E),
        initializer=tf.keras.initializers.glorot_normal(),
        trainable=True)
    self.ff_kernel = self.add_weight(
        name='ff_kernel',
        shape=(1, H, H, E, E),
        initializer=tf.keras.initializers.glorot_normal(),
        trainable=True)

  @staticmethod
  def _call(
      embedding,
      key_kernel,
      query_kernel,
      value_kernel,
      layer_norm1,
      ff_kernel,
      layer_norm2,
      activation,
  ):
    """Run multiheaded attention matrix math.
    
    B = batch size.
    T = number of tokens.
    E = embedding length. 

    Params:
      token: Tensor of shape BHTE (or BTE if first layer).
      K: Kernel of shape 1HE
      Q: Kernel of shape 1HE
      V: Kernel of shape 1HEE
      FF: Kernel of shape 1HHE
    """
    # BHTE, BHE -> BHT
    key = tf.einsum('...TE, ...E -> ...T', embedding, key_kernel)

    # BHTE, BHE -> BHT
    query = tf.einsum('...TE, ...E -> ...T', embedding, query_kernel)

    # BHTE, BHeE -> BHTE
    # This is akin to a depthwise conv2D.
    value = tf.einsum('...TE, ...Ee -> ...Te', embedding, value_kernel)

    # Cross product of the keys and queries to generate indices on each value.
    # BHT, BHt -> BHTt
    # This is the global information mixing that is unlike any
    # existing conv structure. Maxpool and atrous convs
    # are still local in nature, not global.
    # The cost of globality is this quadratic expansion, the Tt.
    kq = tf.einsum('...T, ...t -> ...Tt', query, key)
    index = tf.nn.softmax(kq, axis=3)

    tf.debugging.assert_near(tf.reduce_sum(index[0, 0, 0]), 1.0)

    # Mix the global information, indexed with KxQ.
    # BHTt, BHtE -> BHTE
    z = tf.einsum('...Tt, ...tE -> ...TE', index, value)
    z = embedding + z
    z = layer_norm1(z)

    # Mix the information along the head axis.
    # This is akin to a 1x1 point conv2D.
    # BHTE, BTtEe -> BHte
    y = tf.einsum('...HTE, ...HhEe -> ...hTe', z, ff_kernel)
    y = z + y
    y = layer_norm2(activation(y))

    return y

  def call(self, embedding):
    if self.token_input:
      # Convert BTE to BHTE, where H is 1 but will be broadcast.
      embedding = embedding[:, tf.newaxis, :, :]

    # Token is shape BHTL.
    z = self._call(embedding, self.key_kernel, self.query_kernel,
                   self.value_kernel, self.layer_norm1, self.ff_kernel,
                   self.layer_norm2, self.activation)

    return z


def get_model(model_name, embedding_size=128) -> tf.keras.models.Model:
  x = input = tf.keras.Input(shape=(224, 224, 3))
  x = ImageTokenizer(8, embedding_size)(x)
  for layer in range(6):
    x = MultiHeadedAttention(embedding_size, 6)(x)

  # In a convnet, the penultimate layer is usually a GlobalAveragePooling.
  # The idea is that the position information is no longer important.
  # Similarly, we assume that the relative token location is no longer
  # important.
  x = tf.reduce_mean(x, axis=2)
  x = tf.keras.layers.Flatten()(x)
  y = tf.keras.layers.Dense(10)(x)

  model = tf.keras.Model(inputs=input, outputs=y)
  model.build(input_shape=(None, 224, 224, 3))
  model.summary()

  o = tfa.optimizers.LAMB(weight_decay=1e-7)

  model.compile(
      optimizer=o,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
  )
  return model


def main():
  train, val, test = get_ds(batch_size=32,
                            train_per=85,
                            valid_per=15,
                            shape=(224, 224, 3))
  model = get_model("test_model")

  rlr = tf.keras.callbacks.ReduceLROnPlateau()

  model.fit(train, epochs=100, validation_data=val, callbacks=[rlr])


if __name__ == '__main__':
  main()