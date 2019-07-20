#!/usr/bin/env python
from tf_utils import *
import numpy as np
from attention_net import Encoder


def attention(input_1, train_flag, mask):
	regularizer = tf.contrib.layers.l2_regularizer(1.0)
	with tf.variable_scope('attention_net', regularizer=regularizer):
		sample_encoder = Encoder(num_layers=2, d_model=64, num_heads=2, 
			dff=256, maxlen=14)
		# (batch_size, input_seq_len, 2), not normalized. Need softmax
		feature_output = sample_encoder(input_1,
			training=train_flag, mask=mask)
		output_1 = tf.nn.softmax(feature_output[:,:, 0], axis=1)
		output_2 = tf.nn.softmax(feature_output[:,:, 1], axis=1)
	return tf.concat([output_1, output_2], 1), feature_output


def fc_net(input_1, output_dim, keep_prob, is_training):
	with tf.variable_scope('first_fc'):
		output = output_layer(input_1, 500)
		output = tf.nn.relu(output)
		output = tf.nn.dropout(output, keep_prob)
	with tf.variable_scope('second_fc'):
		output = output_layer(output, 100)
		output = tf.nn.relu(output)
		output = tf.nn.dropout(output, keep_prob)
	# with tf.variable_scope('third_fc'):
	# 	output = output_layer(output, 50)
	# 	output = tf.nn.relu(output)
	# with tf.variable_scope('forth_fc'):
	# 	output = output_layer(output, 100)
	# 	output = tf.nn.relu(output)
	with tf.variable_scope('output'):
		with tf.variable_scope('X'):
			output_1 = tf.nn.softmax(output_layer(output, output_dim/2))
		with tf.variable_scope('Y'):
			output_2 = tf.nn.softmax(output_layer(output, output_dim/2))
	return tf.concat([output_1, output_2], 1), output


def simple_conv(inputs):
    layer1_out = tf.layers.conv1d(inputs=inputs,
        filters=8, kernel_size=10,
        padding='same', data_format='channels_last',
        activation=tf.nn.relu, name='layer1')
    layer2_out = tf.layers.conv1d(inputs=layer1_out,
        filters=32, kernel_size=10,
        padding='same', data_format='channels_last',
        activation=tf.nn.relu, name='layer2')
    layer3_out = tf.layers.conv1d(inputs=layer2_out,
        filters=32, kernel_size=10,
        padding='same', data_format='channels_last',
        activation=tf.nn.relu, name='layer3')
    layer4_out = tf.layers.conv1d(inputs=layer3_out,
        filters=8, kernel_size=10,
        padding='same', data_format='channels_last',
        activation=tf.nn.relu, name='layer4')
    layer5_out = tf.layers.conv1d(inputs=layer4_out,
        filters=2, kernel_size=10,
        padding='same', data_format='channels_last',
        activation=None, name='layer5')
	output_1 = tf.nn.softmax(layer5_out[:,:, 0], axis=1)
	output_2 = tf.nn.softmax(layer5_out[:,:, 1], axis=1)
	return tf.concat([output_1, output_2], 1), layer5_out


def test_graph(train_dir='logs'):
	'''
	Run this function to look at the graph structure on tensorboard. A fast way!
	:param train_dir:
	'''
	input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
	# result = densenet(input_tensor, 10, 40, 12, 1)
	result = se_resnet(input_tensor, 10, 5, 1, True, 16)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	train_writer=tf.summary.FileWriter(train_dir,sess.graph)





