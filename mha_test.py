"""Unit tests for mha module."""

import unittest
from typing import List

import tensorflow as tf
import numpy as np

import mha


class MhaTest(tf.test.TestCase):

  def test_chip_image(self):
    """Tests that the tokenization of an image is correct."""
    image = tf.range((11 * 224 * 224 * 3))
    image = tf.reshape(image, (11, 224, 224, 3))
    chips = mha.ImageTokenizer.chip_image(image, 11, 224, 224, 32)
    self.assertEqual(chips.shape, (11, 7, 7, 32, 32, 3))

  def test_tokenize_image(self):
    """Tests that the tokenization of an image is correct."""
    image = tf.range((11 * 224 * 224 * 3))
    image = tf.reshape(image, (11, 224, 224, 3))
    image_tokenizer = mha.ImageTokenizer(patch_size=32, token_length=512)
    tokens = image_tokenizer(image)

    self.assertEqual(tokens.shape, (11, 49, 512))

  def test_get_model(self):
    """Tests that the model can be built."""
    model = mha.get_model('test_mha_model')
    model.build((11, 224, 224, 3))
    model.summary()