import math
import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

import utils

IMAGE_SIZE = 28
IMAGE_CHANNEL_NUM = 1
CONV_1_SIZE = 6
CONV_1_DEPTH = 6
CONV_2_SIZE = 5
CONV_2_DEPTH = 12
CONV_3_SIZE = 4
CONV_3_DEPTH = 24
FC_SIZE = 200
OUTPUT_SIZE = 10

LEARNING_RATE_MAX = 0.003
LEARNING_RATE_MIN = 0.0001
LEARNING_RATE_DECAY_SPEED = 2000
REGULARIZATION_RATE = 0.0001

SUMMARY_INTERVAL = 50
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 1000

MODEL_NAME = "mnist"


def main():
    parser = build_parser()
    options = parser.parse_args()
    create_if_not_exist(options.model_dir)
    create_if_not_exist(options.log_dir)
    mnist_data = input_data.read_data_sets(options.data_dir, one_hot=True, reshape=False)
    train(mnist_data, options)


def inference(input_tensor, regularizer=None):
    with tf.variable_scope("layer_1_conv"):
        conv_1_weight = utils.get_weight([CONV_1_SIZE, CONV_1_SIZE, IMAGE_CHANNEL_NUM, CONV_1_DEPTH])
        conv_1_bias = utils.get_bias([CONV_1_DEPTH])
        conv_1 = utils.conv2d(input_tensor, conv_1_weight, stride=1)
        conv_1_activation = tf.nn.relu(tf.nn.bias_add(conv_1, conv_1_bias))
    with tf.variable_scope("layer_2_conv"):
        conv_2_weight = utils.get_weight([CONV_2_SIZE, CONV_2_SIZE, CONV_1_DEPTH, CONV_2_DEPTH])
        conv_2_bias = utils.get_bias([CONV_2_DEPTH])
        conv_2 = utils.conv2d(conv_1_activation, conv_2_weight, stride=2)
        conv_2_activation = tf.nn.relu(tf.nn.bias_add(conv_2, conv_2_bias))
    with tf.variable_scope("layer_3_conv"):
        conv_3_weight = utils.get_weight([CONV_3_SIZE, CONV_3_SIZE, CONV_2_DEPTH, CONV_3_DEPTH])
        conv_3_bias = utils.get_bias([CONV_3_DEPTH])
        conv_3 = utils.conv2d(conv_2_activation, conv_3_weight, stride=2)
        conv_3_activation = tf.nn.relu(tf.nn.bias_add(conv_3, conv_3_bias))
        shape = conv_3_activation.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        conv_3_activation_reshaped = tf.reshape(conv_3_activation, [-1, nodes])
    with tf.variable_scope("layer_4_fc"):
        fc_1_weight = utils.get_weight([nodes, FC_SIZE])
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc_1_weight))
        fc_1_bias = utils.get_bias([FC_SIZE])
        fc_1_a = tf.nn.relu(tf.matmul(conv_3_activation_reshaped, fc_1_weight) + fc_1_bias)
    with tf.variable_scope("layer_5_fc"):
        fc_2_weight = utils.get_weight([FC_SIZE, OUTPUT_SIZE])
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc_2_weight))
        fc_2_bias = utils.get_bias([OUTPUT_SIZE])
        logits = tf.matmul(fc_1_a, fc_2_weight) + fc_2_bias
    return logits


def train(mnist_data, options):
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL_NUM], name='x')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y_')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logits = inference(x, regularizer)
    y = tf.nn.softmax(logits, name='output')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    decay = tf.train.exponential_decay((LEARNING_RATE_MAX - LEARNING_RATE_MIN), global_step,
                                       LEARNING_RATE_DECAY_SPEED, math.exp(-1), staircase=False)
    learning_rate = LEARNING_RATE_MIN + decay

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(options.log_dir + '/train', sess.graph)
        validate_writer = tf.summary.FileWriter(options.log_dir + '/validate')
        sess.run(init)
        for i in range(1, options.iterations + 1):
            x_batch, y_batch = mnist_data.train.next_batch(options.batch_size)
            sess.run(train_step, {x: x_batch, y_: y_batch})
            if i % SUMMARY_INTERVAL == 0:
                t_summary, t_accuracy, t_loss, step = \
                    sess.run([merged, accuracy, loss, global_step], feed_dict={x: x_batch, y_: y_batch})
                train_writer.add_summary(t_summary, i)
                v_summary, v_accuracy, v_loss = \
                    sess.run([merged, accuracy, loss],
                             feed_dict={x: mnist_data.validation.images, y_: mnist_data.validation.labels})
                validate_writer.add_summary(v_summary, i)
                if i % PRINT_INTERVAL == 0:
                    print("**** Iteration %d ****" % i)
                    print("Train accuracy = %g, loss = %g" % (t_accuracy, t_loss))
                    print("Validate accuracy = %g, loss = %g" % (v_accuracy, v_loss))
                if i % SAVE_INTERVAL == 0:
                    saver.save(sess, os.path.join(options.model_dir, MODEL_NAME), global_step=step)

        train_writer.close()
        validate_writer.close()

        graph_def = tf.get_default_graph().as_graph_def()
        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
        with tf.gfile.GFile(os.path.join(options.model_dir, MODEL_NAME + '.pb'), 'wb') as f:
            f.write(output_graph.SerializeToString())


def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, dest='data_dir', default='./data',
                        help='directory of MNIST data set')
    parser.add_argument('--model_dir', type=str, dest='model_dir', default='./saved_model',
                        help='directory to save checkpoints and model')
    parser.add_argument('--log_dir', type=str, dest='log_dir', default='./log',
                        help='directory to save logs')
    parser.add_argument('--iterations', type=int, dest='iterations', default=10000,
                        help='training iterations')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=100,
                        help='batch size')
    return parser


if __name__ == '__main__':
    main()
