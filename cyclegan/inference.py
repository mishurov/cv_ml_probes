#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, "../deps/opencv/lib/python3.5/dist-packages")

from absl import flags
import tensorflow as tf
import numpy as np
import cv2

from tensorflow_research import data_provider
from tensorflow_research import networks

flags.DEFINE_string('video', None, 'The path to the video file.')
flags.DEFINE_string('model', None, 'The path to the training checkpoint.')
flags.DEFINE_integer('size', 256, 'The patch size of images.')
flags.DEFINE_integer('resnet', 6, 'The number of Resnet blocks.')

FLAGS = flags.FLAGS

TEST_PATH = "training/horse2zebra/testA"
TEST_SET = [
    "n02381460_490.jpg",
    "n02381460_7230.jpg",
    "n02381460_510.jpg",
    "n02381460_4530.jpg",
    "n02381460_800.jpg",
    "n02381460_8980.jpg",
    "n02381460_9260.jpg",
    "n02381460_4660.jpg",
    "n02381460_1110.jpg",
]

# image grid constants
ROWS = 3
GAP = 25
PAIR_GAP = 3
BORDER = 15
CANVAS_COLOR = 255, 255, 255

WINDOW_NAME = "CycleGAN"


def make_inference_graph(model_name, patch_dim, num_resnet_blocks=6):
  input_hwc_pl = tf.placeholder(tf.float32, [None, None, 3])

  images_x = tf.expand_dims(
      data_provider.full_image_to_patch(
          input_hwc_pl, patch_dim, training=False), 0)

  with tf.variable_scope(model_name):
    with tf.variable_scope('Generator'):
      generated = networks.generator(images_x, num_resnet_blocks)
  return input_hwc_pl, generated


def get_canvas_size():
  size = FLAGS.size
  test_len = len(TEST_SET)
  row_len = min(ROWS, test_len)
  w = row_len * size * 2
  w += row_len * PAIR_GAP
  w += (row_len - 1) * GAP
  w += BORDER * 2

  col_len = (test_len - 1) // ROWS + 1
  h = col_len * size
  h += (col_len - 1) * GAP
  h += BORDER * 2
  return w, h


def get_solid(w, h):
  shape = h, w, 3
  return np.full(shape, CANVAS_COLOR, dtype=np.uint8)


def crop_resize(img):
  size = FLAGS.size
  h, w, _ = img.shape
  if w == size and h == size:
    return img
  min_dim = min(w, h)
  x = (w - min_dim) // 2
  y = (h - min_dim) // 2
  img = img[y:y + h, x:x + w]
  return cv2.resize(img, (size, size))


def get_output_img(input_img, sess, input_pl, output_tensor):
  input_np = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
  output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
  return data_provider.undo_normalize_image(output_np)


def get_pair(input_img, sess, input_pl, output_tensor):
  size = FLAGS.size
  gap = get_solid(PAIR_GAP, size)
  output_img = get_output_img(input_img, sess, input_pl, output_tensor)
  input_img = crop_resize(input_img)
  return np.hstack((input_img, gap, output_img))


def show_grid(sess, input_pl, output_tensor):
  size = FLAGS.size
  canvas_w, canvas_h = get_canvas_size()
  img = get_solid(canvas_w, canvas_h)
  for i, file_name in enumerate(TEST_SET):
    file_path = os.path.join(TEST_PATH, file_name)
    input_img = cv2.imread(file_path)
    print("processing test image {}".format(i + 1))
    pair = get_pair(input_img, sess, input_pl, output_tensor)
    pair_w = size * 2 + PAIR_GAP
    n = i // ROWS
    m = i - n * ROWS
    offset_x = BORDER + m * pair_w + m * GAP
    offset_y = BORDER + n * size + n * GAP
    img[offset_y:offset_y + size, offset_x:offset_x + pair_w] = pair

  cv2.imshow(WINDOW_NAME, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def show_video(file_path, sess, input_pl, output_tensor):
  size = FLAGS.size
  cap = cv2.VideoCapture(file_path)
  ret = True
  total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frames = []
  i = 1
  while(True):
    ret, frame = cap.read()
    if not ret:
      break
    print("processing frame {} of {}".format(i, total), end='\r')
    i += 1
    input_img = crop_resize(frame)
    output_img = get_output_img(input_img, sess, input_pl, output_tensor)

    scale = int(size * 1.5)
    input_img = cv2.resize(input_img, (scale, scale))
    output_img = cv2.resize(output_img, (scale, scale))
    gap = get_solid(PAIR_GAP, scale)
    pair = np.hstack((input_img, gap, output_img))
    frames.append(pair)

  i = -1
  while(True):
    i = i + 1 if i < len(frames) - 1 else 0
    cv2.imshow(WINDOW_NAME, frames[i])
    if cv2.waitKey(30) in [ord("q"), 27]:
        break


def main(_):
  size = FLAGS.size
  resnet = FLAGS.resnet
  model = FLAGS.model
  video_path = FLAGS.video
  images_x_hwc_pl, generated_y = make_inference_graph('ModelX2Y', size, resnet)
  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    saver.restore(sess, model)
    if video_path:
      show_video(video_path, sess, images_x_hwc_pl, generated_y)
    else:
      show_grid(sess, images_x_hwc_pl, generated_y)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('model')
  tf.app.run()
