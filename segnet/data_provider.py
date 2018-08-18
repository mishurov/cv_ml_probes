from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def replace_ignore_label(label, ignore_label=255):
  cond = tf.equal(label, ignore_label)
  cond_true = tf.constant(-1, shape=label.get_shape())
  cond_false = label
  replaced_label = tf.where(cond, cond_true, cond_false)
  return replaced_label


def randomly_scale(image, label, output_shape,
                   min_scale=1.0, max_scale=1.5, ignore_label=255):
  bx = tf.expand_dims(image, 0)
  by = tf.expand_dims(label, 0)

  by = tf.to_int32(by)

  input_shape = tf.to_float(tf.shape(bx)[1:3])
  scale_shape = output_shape / input_shape

  rand_var = tf.random_uniform(shape=[1], minval=min_scale, maxval=max_scale)
  scale = tf.reduce_min(scale_shape) * rand_var

  scaled_input_shape = tf.to_int32(tf.round(input_shape * scale))

  resized_image = tf.image.resize_nearest_neighbor(bx, scaled_input_shape)
  resized_label = tf.image.resize_nearest_neighbor(by, scaled_input_shape)

  resized_image = tf.squeeze(resized_image, axis=0)
  resized_label = tf.squeeze(resized_label, axis=0)

  shifted_classes = resized_label + 1

  cropped_image = tf.image.resize_image_with_crop_or_pad(
      resized_image, output_shape[0], output_shape[1])

  cropped_label = tf.image.resize_image_with_crop_or_pad(
      shifted_classes, output_shape[0], output_shape[1])

  mask = tf.to_int32(tf.equal(cropped_label, 0)) * (ignore_label + 1)
  cropped_label = cropped_label + mask - 1

  return cropped_image, cropped_label


def get_image_tensor(filename, channels=3):
  image_string = tf.read_file(filename)
  return tf.image.decode_png(image_string, channels=channels)


def parse_text_line(line, path):
  split = tf.string_split([line])
  image_filename = split.values[0]
  label_filename = split.values[1]
  pattern = "\/SegNet"
  image_filename = tf.regex_replace(image_filename, pattern, path)
  label_filename = tf.regex_replace(label_filename, pattern, path)

  image = get_image_tensor(image_filename, channels=3)
  label = get_image_tensor(label_filename, channels=1)

  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.int32)
  image = image / 255.

  target_height = FLAGS.height
  target_width = FLAGS.width
  resized_image, resized_label = randomly_scale(
      image, label, [target_height, target_width])
  resized_label = replace_ignore_label(resized_label)
  return {'image': resized_image}, resized_label


def input_fn(path, batch_size=1):
  train_txt = path + "/CamVid/train.txt"
  dataset = tf.data.TextLineDataset(train_txt)
  dataset = dataset.shuffle(367)
  dataset = dataset.repeat()
  dataset = dataset.map(lambda i: parse_text_line(i, path))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  with tf.name_scope("input"):
    return dataset.make_one_shot_iterator().get_next()
