"""Ground up implementation of multi-headed attention."""

from typing import Tuple
import os
import sys

# Quiet down TF's logging.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

# Force lazy malloc of GPU memory.
physical_devices = tf.config.list_physical_devices("GPU")
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
RANDGEN = tf.random.Generator.from_non_deterministic_state()

NCLASSES = 10
BATCHSIZE = 512


def _alexnet_resize_one_hot(img, label):
    """Alexnet preprocessing.

    Resize 1:1 so that the short edge is 256, then center crop to 256x256.

    Args:
      img: The image tensor. A single example, not a batch.
      label: The label tensor as an integer.

    Returns:
      Tuple[tf.Tensor, tf.Tensor]: The image and label tensors
    """
    # Validate args
    tf.debugging.assert_rank(img, 3)

    orig_hwc = tf.shape(img)
    if orig_hwc[0] > orig_hwc[1]:
        img = tf.transpose(img, [1, 0, 2])

    hwc = tf.cast(tf.shape(img), tf.float32)
    new_h = 256
    new_w = tf.cast(hwc[1] / hwc[0] * 256.0, tf.int32)

    img = tf.image.resize(img, (new_h, new_w))

    x = new_w - 256
    img = img[:, x // 2 : x // 2 + 256, :]

    if orig_hwc[0] > orig_hwc[1]:
        img = tf.transpose(img, [1, 0, 2])

    return img, tf.one_hot(label, 10, dtype=tf.float32)


def _alexnet_crop_train(img):
    """Alexnet augmentation for train.

    Take random 224x224 crops.

    Args:
      img: The image tensor. A single example, not a batch.

    Returns:
      tf.Tensor: The image tensor.
    """
    y = RANDGEN.uniform(shape=(), minval=0, maxval=32, dtype=tf.int32)
    x = RANDGEN.uniform(shape=(), minval=0, maxval=32, dtype=tf.int32)
    return img[y : y + 224, x : x + 224, :]


def _alexnet_crop_valid_center(img):
    """Simplified version of AlexNet validation cropping.

    The original aglorithm takes 5 crops: 4 corners and center. This
    algo just takes the center, to reduce validation inference time.

    Args:
      img: The image tensor. A single example, not a batch.

    Returns:
      tf.Tensor: The image tensor.
    """
    return img[16:240, 16:240, :]


def get_ds(
    batch_size: int,
    train_per: int,
    valid_per: int,
    shape: Tuple[int, int, int] = (224, 224, 3),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Creates the Imagenette dataset.

    Args:
      batch_size (int): The batch size.
      train_per (int): The percentage of the train set to use.
      valid_per (intr): The percentage of the validation set to use.
      shape (Tuple[int, int, int]): The shape of the image as HWC.

    Returns:
      Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The train, validation, and test datasets.
    """
    datasets = tfds.load(
        "imagenette",
        split=[f"train[:{train_per}%]", f"validation[:{valid_per}%]"],
        shuffle_files=True,
        with_info=False,
        as_supervised=True,
        download=True,
    )

    dataset_train, dataset_validation = datasets

    dataset_train = dataset_train.map(
        lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label),
        num_parallel_calls=AUTOTUNE,
    )
    dataset_validation = dataset_validation.map(
        lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label),
        num_parallel_calls=AUTOTUNE,
    )

    assert shape[0] == 224

    dataset_train = dataset_train.map(
        lambda img, label: _alexnet_resize_one_hot(img, label),
        num_parallel_calls=AUTOTUNE,
    )
    dataset_validation = dataset_validation.map(
        lambda img, label: _alexnet_resize_one_hot(img, label),
        num_parallel_calls=AUTOTUNE,
    )

    dataset_train = dataset_train.cache(
        f"/tmp/cached_imagenette_train_{train_per}_alexnet"
    )

    dataset_train = dataset_train.shuffle(1000)
    # No shuffle for validation.

    dataset_train = dataset_train.map(
        lambda img, label: (_alexnet_crop_train(img), label),
        num_parallel_calls=AUTOTUNE,
    )
    dataset_validation = dataset_validation.map(
        lambda img, label: (_alexnet_crop_valid_center(img), label),
        num_parallel_calls=AUTOTUNE,
    )

    dataset_train = dataset_train.batch(batch_size, drop_remainder=False)
    dataset_validation = dataset_validation.batch(batch_size, drop_remainder=False)

    dataset_train = dataset_train.prefetch(AUTOTUNE)
    dataset_validation = dataset_validation.prefetch(AUTOTUNE)

    nvalid = len(dataset_validation)
    assert nvalid >= 2

    dataset_test = dataset_validation.skip(nvalid // 2)
    dataset_validation = dataset_validation.take(nvalid // 2)

    return dataset_train, dataset_validation, dataset_test


class ImageTokenizer(tf.keras.layers.Layer):
    """Image tokenizer layer.

    This layer takes an image and turns it into a sequence of embeddings.

    Args:
      patch_size(int): The size of the patches to break the image into.
      token_length(int): The length of the token embeddings.
    """

    def __init__(self, patch_size: int, token_length: int):
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
        """Turns an image into a sequence of embeddings.

        Includes both positional and image embeddings.

        Args:
          image: A tensor of shape [batch_size, height, width, 3].
          B: Batch size.
          H: Height of image.
          W: Width of image.
          P: Patch size, e.g. 32x32.

        Returns:
          A tensor of shape [batch_size, height // patch_size, width // patch_size,
          patch_size, patch_size, 3].
        """
        tf.debugging.assert_equal(H, W)

        vectorized = True

        if vectorized == False:
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

    ### This is where the embeddings are created from the chips.
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
    """Multi-headed attention layer.

    Implemented from the ground up using tf.einsum.

    Args:
      embedding_length(int): The length of the token embeddings.
      num_heads(int): The number of heads to use.
    """

    def __init__(self, embedding_length: int, num_heads: int):
        super().__init__()

        self.embedding_length = embedding_length
        self.num_heads = num_heads

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.activation = tf.keras.layers.Activation("gelu")

    def build(self, input_shape):
        if len(input_shape) == 3:
            # This is the first layer, so we need to create a new axis
            # for the Heads dimension.
            self.token_input = True
        elif len(input_shape) == 4:
            # BHTD
            self.token_input = False
        else:
            assert False

        self.B = input_shape[0]
        self.T = input_shape[1]

        H = self.num_heads
        E = self.embedding_length

        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=(1, H, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=(1, H, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=(1, H, E, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

        self.value_bias = self.add_weight(
            name="value_bias",
            shape=(1, H, 1, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

        self.ff_kernel = self.add_weight(
            name="ff_kernel",
            shape=(1, H, H, E, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

        self.ff_bias = self.add_weight(
            name="ff_bias",
            shape=(1, H, 1, E),
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True,
        )

    @staticmethod
    def _call(
        embedding,
        key_kernel,
        query_kernel,
        value_kernel,
        value_bias,
        layer_norm1,
        ff_kernel,
        ff_bias,
        layer_norm2,
        activation,
    ):
        """Run multiheaded attention matrix math.

        Takes a batch of embeddings and runs multiheaded attention on them.

        First, keys, queries, and values are computed from the embedding.
        Then, the attention matrix is computed as a outer product of K and Q.
        The attention matrix is what mixes information between tokens.

        The attention matrix is then softmaxed so that it acts as a
        gate or index into the newly generated Values.

        that value, now called Z, is added to the embedding, and then
        passed through a feed forward network that mixes the information
        at each token location between heads, aking to a 1x1 pointwise
        convolution. Finally, bias, activation, and normalization are applied.

        B = batch size.
        H = number of heads.
        T = number of tokens / embeddings.
        D = dimension of the token / embeddings, aka length of the vector.

        Params:
          embedding: BTD or BHTD
          key_kernel: 1HD
          query_kernel: 1HD
          value_kernel: 1HdD
          value_bias: 1H1D
          layer_norm1:
          ff_kernel: 1HHdD
          ff_bias: 1H1D
          layer_norm2:
          activation: activation to run at the end of the feed forward.

        Returns:
          New embeddings of shape BHTD.
        """
        # BHTD, BHE -> BHT
        key = tf.einsum("...TD, ...D -> ...T", embedding, key_kernel)

        # BHTD, BHE -> BHT
        query = tf.einsum("...TD, ...D -> ...T", embedding, query_kernel)

        # BHTD, BHeE -> BHTD
        value = tf.einsum("...TD, ...Dd -> ...TD", embedding, value_kernel)
        value = value + value_bias

        # Cross product of the keys and queries to generate indices on each value.
        #
        # This is the heart of the information mixing that is unlike any
        # existing conv structure. Information from anywhere in the image,
        # within the same head, is mixed in. Note that the depth of
        # the embedding (typically 512) is somewhat akin to the number of
        # channels in a convolution. By comparison, maxpool and atrous convs are
        # only able to mix information across an image after multiple layers.
        # The cost of globality is this quadratic expansion, the Tt.
        #
        # BHT, BHt -> BHTt
        kq = tf.einsum("...T, ...t -> ...Tt", query, key)

        # Label this as index, because it is a gate or index into the Values.
        # A softmax (and sigmoid) acting as a gate is a common design
        # pattern in deep learning, such as the gates in an LSTM.
        #
        # BHTt, where summing along t now yields 1.0 as a probability distribution.
        index = tf.nn.softmax(kq, axis=3)

        tf.debugging.assert_near(tf.reduce_sum(index[0, 0, 0]), 1.0)

        # Apply the index (or probability distribution) by multiplying
        # the softmaxed values t and summing.
        # BHTt, BHtD -> BHTD
        z = tf.einsum("...Tt, ...tD -> ...TD", index, value)

        # Create a residual skip connections and then norm.
        #
        # BHTD, BHTD -> BHTD
        z = embedding + z
        z = layer_norm1(z)

        # Mix the information along the head and embedding axis, holding
        # the token axis fixed.
        # This is akin to a 1x1 point conv2D.
        #
        # BHTD, BHhEe -> BhTd
        y = tf.einsum("...HTD, ...HhDd -> ...hTd", z, ff_kernel)
        y = y + ff_bias

        # Create a residual skip connection, activate, and then norm.
        y = z + y
        y = layer_norm2(activation(y))

        return y

    def call(self, embedding):
        if self.token_input:
            # Convert BTD to BHTD, where H is 1 but will be broadcast.
            embedding = embedding[:, tf.newaxis, :, :]

        # Token is shape BHTL.
        z = self._call(
            embedding,
            self.key_kernel,
            self.query_kernel,
            self.value_kernel,
            self.value_bias,
            self.layer_norm1,
            self.ff_kernel,
            self.ff_bias,
            self.layer_norm2,
            self.activation,
        )

        return z


def get_model(model_name, embedding_size=64) -> tf.keras.models.Model:
    x = input = tf.keras.Input(shape=(224, 224, 3))
    x = ImageTokenizer(16, embedding_size)(x)
    for layer in range(4):
        x = MultiHeadedAttention(embedding_size, 4)(x)

    # In early convnets, the penultimate layer was is usually a Flatten,
    # making the final layer a linear projection.
    # A more modern approach is a GlobalAveragePooling layer.
    # The idea is that the position information is no longer important,
    # it's the existence of features that matters.
    #
    # In our case, an embedding is 128 or 256 features. Should we
    # mean across heads, tokens, or embedding dimension? The embedding
    # is analogous to filters so we definitely don't want to lose that.
    # The Heads is a reasonable guess but it's very small. Let's try
    # reducing along the tokens dimension for now, and experiment
    # with other reductions later.
    x = tf.reduce_mean(x, axis=2)  # BHTD -> BHE
    x = tf.keras.layers.Flatten()(x)
    y = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs=input, outputs=y)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    o = tfa.optimizers.LAMB(weight_decay=1e-5)

    model.compile(
        optimizer=o,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model


def main():
    train, val, test = get_ds(
        batch_size=32, train_per=85, valid_per=15, shape=(224, 224, 3)
    )
    model = get_model("test_model")

    rlr = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)

    model.fit(train, epochs=100, validation_data=val, callbacks=[rlr])

    return 0


if __name__ == "__main__":
    sys.exit(main())
