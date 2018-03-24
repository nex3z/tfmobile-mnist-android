import tensorflow as tf
import os
from argparse import ArgumentParser

from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
import math
import utils

IMAGE_SIZE = 28
IMAGE_CHANNEL_NUM = 1
CONV_1_SIZE = 6
CONV_1_DEPTH = 24
CONV_2_SIZE = 5
CONV_2_DEPTH = 48
CONV_3_SIZE = 4
CONV_3_DEPTH = 64
FC_1_SIZE = 200
OUTPUT_SIZE = 10

BATCH_NORM_EPSILON = 1e-5
LEARNING_RATE_MAX = 0.02
LEARNING_RATE_MIN = 0.0001
LEARNING_RATE_DECAY_SPEED = 1600
KEEP_PROB_CONV = 1.0
KEEP_PROB_FC = 0.75

SUMMARY_INTERVAL = 50
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 1000

MODEL_NAME = "mnist_bn"


def main():
    parser = build_parser()
    options = parser.parse_args()
    utils.create_if_not_exist(options.model_dir)
    utils.create_if_not_exist(options.log_dir)
    mnist_data = input_data.read_data_sets(options.data_dir, one_hot=True, reshape=False)
    train(mnist_data, options)


def batch_norm(logits, offset, iteration, is_test, is_conv=True):
    ema = tf.train.ExponentialMovingAverage(0.999, iteration)
    if is_conv:
        mean, variance = tf.nn.moments(logits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(logits, [0])
    update_ema = ema.apply([mean, variance])
    m = tf.cond(is_test, lambda: ema.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: ema.average(variance), lambda: variance)
    bn = tf.nn.batch_normalization(logits, m, v, offset, None, BATCH_NORM_EPSILON)
    return bn, update_ema


def inference(x, is_test):
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('layer_1_conv', reuse=tf.AUTO_REUSE):
        conv_1_weight = utils.get_weight([CONV_1_SIZE, CONV_1_SIZE, IMAGE_CHANNEL_NUM, CONV_1_DEPTH])
        conv_1_bias = utils.get_bias([CONV_1_DEPTH])
        conv_1 = utils.conv2d(x, conv_1_weight, stride=1)
        conv_1_bn, update_ema_1 = batch_norm(conv_1, conv_1_bias, global_step, is_test)
        conv_1_a = tf.nn.relu(conv_1_bn)
        conv_1_out = tf.cond(is_test, lambda: conv_1_a, lambda: utils.drop_out(conv_1_a, KEEP_PROB_CONV))

    with tf.variable_scope('layer_2_conv', reuse=tf.AUTO_REUSE):
        conv_2_weight = utils.get_weight([CONV_2_SIZE, CONV_2_SIZE, CONV_1_DEPTH, CONV_2_DEPTH])
        conv_2_bias = utils.get_bias([CONV_2_DEPTH])
        conv_2 = utils.conv2d(conv_1_out, conv_2_weight, stride=2)
        conv_2_bn, update_ema_2 = batch_norm(conv_2, conv_2_bias, global_step, is_test)
        conv_2_a = tf.nn.relu(conv_2_bn)
        conv_2_out = tf.cond(is_test, lambda: conv_2_a, lambda: utils.drop_out(conv_2_a, KEEP_PROB_CONV))

    with tf.variable_scope('layer_3_conv', reuse=tf.AUTO_REUSE):
        conv_3_weight = utils.get_weight([CONV_3_SIZE, CONV_3_SIZE, CONV_2_DEPTH, CONV_3_DEPTH])
        conv_3_bias = utils.get_bias([CONV_3_DEPTH])
        conv_3 = utils.conv2d(conv_2_out, conv_3_weight, stride=2)
        conv_3_bn, update_ema_3 = batch_norm(conv_3, conv_3_bias, global_step, is_test)
        conv_3_a = tf.nn.relu(conv_3_bn)
        conv_3_out = tf.cond(is_test, lambda: conv_3_a, lambda: utils.drop_out(conv_3_a, KEEP_PROB_CONV))
        shape = conv_3_out.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        conv_3_reshaped = tf.reshape(conv_3_out, [-1, nodes])

    with tf.variable_scope('layer_4_fc', reuse=tf.AUTO_REUSE):
        fc_1_weight = utils.get_weight([nodes, FC_1_SIZE])
        fc_1_bias = utils.get_bias([FC_1_SIZE])
        fc_1 = tf.matmul(conv_3_reshaped, fc_1_weight)
        fc_1_bn, update_ema_4 = batch_norm(fc_1, fc_1_bias, global_step, is_test, is_conv=False)
        fc_1_a = tf.nn.relu(fc_1_bn)
        fc_1_out = tf.cond(is_test, lambda: fc_1_a, lambda: tf.nn.dropout(fc_1_a, KEEP_PROB_FC))

    with tf.variable_scope('layer_5_fc', reuse=tf.AUTO_REUSE):
        fc_2_weight = utils.get_weight([FC_1_SIZE, OUTPUT_SIZE])
        fc_2_bias = utils.get_bias([OUTPUT_SIZE])
        logits = tf.matmul(fc_1_out, fc_2_weight) + fc_2_bias

    update_ema = tf.group(update_ema_1, update_ema_2, update_ema_3, update_ema_4)

    return logits, global_step, update_ema


def train(mnist_data, options):
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL_NUM], name='x')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y_')
    is_test = tf.placeholder_with_default(True, shape=[])

    logits, global_step, update_ema = inference(x, is_test)
    y = tf.nn.softmax(logits, name='output')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('loss', loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    decay = tf.train.exponential_decay((LEARNING_RATE_MAX - LEARNING_RATE_MIN), global_step,
                                       LEARNING_RATE_DECAY_SPEED, math.exp(-1), staircase=False)
    learning_rate_tensor = LEARNING_RATE_MIN + decay

    train_step = tf.train.AdamOptimizer(learning_rate_tensor).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, update_ema]):
        train_op = tf.no_op(name='train')

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(options.log_dir, 'train'), sess.graph)
        validate_writer = tf.summary.FileWriter(os.path.join(options.log_dir, 'validate'))
        sess.run(init)
        for i in range(1, options.iterations + 1):
            x_batch, y_batch = mnist_data.train.next_batch(options.batch_size)
            sess.run(train_op, {x: x_batch, y_: y_batch, is_test: False})
            if i % SUMMARY_INTERVAL == 0:
                t_summary, t_accuracy, t_loss, step = \
                    sess.run([merged, accuracy, loss, global_step], feed_dict={x: x_batch, y_: y_batch, is_test: False})
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
        with tf.gfile.GFile(os.path.join(options.model_dir, MODEL_NAME+'.pb'), 'wb') as f:
            f.write(output_graph.SerializeToString())


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
