"""Unit tests for mha module."""

import tensorflow as tf

import mha


class MhaTest(tf.test.TestCase):
    def test_chip_image(self):
        """Tests that the chipping of an image is correct."""
        # Arrange
        image = tf.range((11 * 224 * 224 * 3))
        image = tf.reshape(image, (11, 224, 224, 3))

        # Act
        chips = mha.ImageTokenizer.chip_image(image, 11, 224, 224, 32)

        # Assert
        self.assertEqual(chips.shape, (11, 7 * 7, 32 * 32 * 3))

    def test_tokenize_image(self):
        """Tests that the tokenization of an image is correct."""
        # Arrange
        image = tf.range((11 * 224 * 224 * 3))
        image = tf.reshape(image, (11, 224, 224, 3))
        image_tokenizer = mha.ImageTokenizer(patch_size=32, token_length=512)

        # Act
        tokens = image_tokenizer(image)

        # Assert
        self.assertEqual(tokens.shape, (11, 49, 512))

    def test_get_model(self):
        """Tests that the model can be built."""
        # Arrange
        model = mha.get_model("test_mha_model")

        # Act
        model.build((11, 224, 224, 3))

        # Assert
        return
