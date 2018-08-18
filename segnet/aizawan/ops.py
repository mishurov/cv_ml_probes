""" ops.py
"""

import numpy as np
import tensorflow as tf


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
    return tf.contrib.layers.xavier_initializer(
        uniform=uniform, seed=seed, dtype=dtype)


def conv2d(incoming, num_filters, filter_size, stride=1, pad='SAME',
           activation=tf.identity,
           weight_init=xavier_initializer(),
           reuse=False, name="conv2d"):
    return tf.layers.conv2d(
        incoming,
        num_filters,
        kernel_size=(filter_size, filter_size),
        strides=(stride, stride),
        padding=pad,
        activation=tf.identity,
        kernel_initializer=weight_init,
        name=name
    )


def maxpool2d(incoming, pool_size, stride=2, pad='SAME', name="maxpool2d"):
    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        pooled = tf.nn.max_pool(x, filter_shape, strides, pad)

    return pooled


def maxpool2d_with_argmax(incoming, pool_size=2, stride=2,
                          name='maxpool_with_argmax'):
    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        _, mask = tf.nn.max_pool_with_argmax(
            x, ksize=filter_shape, strides=strides, padding='SAME')
        mask = tf.stop_gradient(mask)

        pooled = tf.nn.max_pool(
            x, ksize=filter_shape, strides=strides, padding='SAME')

    return pooled, mask


def upsample(incoming, size, name='upsample'):
    x = incoming
    with tf.name_scope(name):
        resized = tf.image.resize_nearest_neighbor(x, size=size)
    return resized


# https://github.com/Pepslee/tensorflow-contrib/blob/master/unpooling.py
def maxunpool2d(incoming, mask, stride=2, name='unpool'):
    x = incoming

    input_shape = incoming.get_shape().as_list()
    strides = [1, stride, stride, 1]
    output_shape = (input_shape[0],
                    input_shape[1] * strides[1],
                    input_shape[2] * strides[2],
                    input_shape[3])

    flat_output_shape = [output_shape[0], np.prod(output_shape[1:])]
    with tf.name_scope(name):
        flat_input_size = tf.size(x)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype),
                                 shape=[output_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b, mask_], 1)

        x_ = tf.reshape(x, [flat_input_size])
        ret = tf.scatter_nd(mask_, x_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def batch_norm(incoming, phase_train, name='batch_norm'):
    return tf.layers.batch_normalization(
        incoming,
        renorm=True,
        momentum=0.95,
        renorm_momentum=0.95,
        gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002),
        training=phase_train,
        trainable=True,
        reuse=False,
        name=name)


def relu(incoming, summary=False, name='relu'):
    x = incoming
    with tf.name_scope(name):
        output = tf.nn.relu(x)
    return output
