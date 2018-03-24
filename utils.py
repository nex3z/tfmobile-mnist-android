import os

import tensorflow as tf


def get_weight(shape):
    return tf.get_variable("weight", shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))


def conv2d(input_tensor, weight, stride):
    return tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding="SAME")


def drop_out(activations, keep_prob):
    return tf.nn.dropout(activations, keep_prob, compatible_conv_noise_shape(activations))


def compatible_conv_noise_shape(y):
    noise_shape = tf.shape(y)
    return noise_shape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])


def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
