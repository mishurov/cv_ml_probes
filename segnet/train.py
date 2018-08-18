from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

import tensorflow as tf
import numpy as np

from data_provider import input_fn
from aizawan import segnet


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_class', 12, 'Number of class to classify')
flags.DEFINE_integer('height', 224, 'Input height')
flags.DEFINE_integer('width', 224, 'Input width')
flags.DEFINE_integer('batch_size', 3, 'Batch size')
flags.DEFINE_integer('steps', 16000, 'Number of training iterations')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_float('learning_rate', 1.0, 'learning rate')

flags.DEFINE_integer('print_step', 100, 'Number of step to print training log')

flags.DEFINE_string('train_set_loc', 'training',
                    'Location with the training set')
flags.DEFINE_string('train_log_dir', 'training/checkpoints',
                    'Directory where to write event logs')
flags.DEFINE_string('export_dir', 'training/export', 'Export directory')

flags.DEFINE_boolean('freeze', False, 'Whether to freeze the model')


FLAGS = flags.FLAGS


def freeze_model(saved_model_path):
  # freeze_graph --input_saved_model_dir=${DIR} \
  #   --output_node_names=scores \
  #   --output_graph=./frozen.pb

  output_node_names = "preds,probs"
  output_graph = saved_model_path + b"/frozen_model.pb"
  cmd = ["freeze_graph",
         "--input_saved_model_dir", saved_model_path,
         "--output_node_names", output_node_names,
         "--output_graph", output_graph]

  tf.logging.info("Freezing model.")
  subprocess.call(cmd)


def export_model(est):
  image = tf.placeholder(
      tf.float32,
      [None, FLAGS.height, FLAGS.width, 3],
      name="image"
  )
  serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
      {'image': image},
      default_batch_size=1
  )

  saved_model_path = est.export_savedmodel(FLAGS.export_dir, serving_input_receiver_fn)

  if FLAGS.freeze:
    freeze_model(saved_model_path)


def segnet_model(features, labels, mode, params):
  inputs = features['image']
  phase_train = mode == tf.estimator.ModeKeys.TRAIN
  logits = segnet.inference(inputs, phase_train)
  probs, preds = segnet.predict(logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    with tf.name_scope("train"):
      loss = segnet.loss(logits, labels)
      acc = segnet.acc(logits, labels)

      tf.summary.scalar('accuracy', acc)

      tf.summary.image('image', inputs)

      labels_normalized = tf.cast(labels, tf.float32)
      labels_normalized = labels_normalized / FLAGS.num_class
      tf.summary.image('labels', labels_normalized)

      shape = preds.get_shape().as_list()
      preds = tf.reshape(preds, [shape[0], shape[1], shape[2], 1])
      preds = tf.cast(preds, tf.float32)
      preds = preds / FLAGS.num_class
      tf.summary.image('predictions', preds)

      optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
      global_step = tf.train.get_or_create_global_step()

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'probs': probs, 'preds': preds}
    export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                      export_outputs=export_outputs)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  run_config = tf.estimator.RunConfig(save_summary_steps=FLAGS.print_step)

  segnet_est = tf.estimator.Estimator(
      model_fn=segnet_model,
      params={},
      model_dir=FLAGS.train_log_dir,
      config=run_config
  )
  segnet_est.train(
      input_fn=lambda: input_fn(FLAGS.train_set_loc, batch_size=FLAGS.batch_size),
      steps=FLAGS.steps
  )

  export_model(segnet_est)


if __name__ == '__main__':
  tf.app.run()
