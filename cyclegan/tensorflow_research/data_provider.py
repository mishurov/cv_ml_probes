# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains code for loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def normalize_image(image):
  """Rescale from range [0, 255] to [-1, 1]."""
  return (tf.to_float(image) - 127.5) / 127.5


def undo_normalize_image(normalized_image):
  """Convert to a numpy array that can be read by PIL."""
  # Convert from NHWC to HWC.
  normalized_image = np.squeeze(normalized_image, axis=0)
  return np.uint8(normalized_image * 127.5 + 127.5)


def _sample_patch(image, patch_size, training=True):
  """Crop image to square shape and resize it to `patch_size`.

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_size: A Python scalar.  The output image size.

  Returns:
    A 3D `Tensor` of HWC format which has the shape of
    [patch_size, patch_size, 3].
  """
  image_shape = tf.shape(image)
  height, width = image_shape[0], image_shape[1]
  target_size = tf.minimum(height, width)
  image = tf.image.resize_image_with_crop_or_pad(image, target_size,
                                                 target_size)
  # tf.image.resize_area only accepts 4D tensor, so expand dims first.
  image = tf.expand_dims(image, axis=0)

  if training:
    scale_size = int(patch_size * 1.12)
    image = tf.image.resize_images(image, [scale_size, scale_size])
    image = tf.squeeze(image, axis=0)
    seed = 10
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.random_crop(
        image, [patch_size, patch_size, image_shape[2]], seed=seed)
  else:
    image = tf.image.resize_images(image, [patch_size, patch_size])
    image = tf.squeeze(image, axis=0)

  # Force image num_channels = 3
  image = tf.tile(image, [1, 1, tf.maximum(1, 4 - tf.shape(image)[2])])
  image = tf.slice(image, [0, 0, 0], [patch_size, patch_size, 3])
  return image


def full_image_to_patch(image, patch_size, training=True):
  image = normalize_image(image)
  # Sample a patch of fixed size.
  image_patch = _sample_patch(image, patch_size, training=training)
  image_patch.shape.assert_is_compatible_with([patch_size, patch_size, 3])
  return image_patch


def parse_dataset(filename, patch_size):
  image_string = tf.read_file(filename)
  image_bytes = tf.image.decode_image(image_string)
  image_patch = full_image_to_patch(image_bytes, patch_size)
  return image_patch


def provide_custom_datasets(image_file_patterns,
                            batch_size,
                            shuffle=True,
                            num_threads=1,
                            patch_size=128):

  outputs = []

  for p in image_file_patterns:
    filenames = tf.gfile.Glob(p)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda f: parse_dataset(f, patch_size))
    dataset = dataset.shuffle(1500, seed=5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    im = dataset.make_one_shot_iterator().get_next()
    outputs.append(im)
  return outputs
