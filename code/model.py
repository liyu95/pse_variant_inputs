#!/usr/bin/env python
from tf_utils import *
import numpy as np
from attention_net import Encoder


def attention():
	regularizer = tf.contrib.layers.l2_regularizer(1.0)
	with tf.variable_scope('attention_net', regularizer=regularizer):
		sample_encoder = Encoder(num_layers=2, d_model=64, num_heads=8, 
			dff=128, maxlen=14)
		self.feature_output = sample_encoder(self.input_x_vector,
			training=self.train_flag, mask=self.mask)

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





def convnet(input_1, input_2, output_dim, keep_prob, is_training):
	with tf.variable_scope('first_fc'):
		output_1 = output_layer(input_1, 10)
		output_1 = tf.nn.relu(output_1)
		# output = tf.nn.dropout(output, keep_prob)

	with tf.variable_scope('snp_conv'):
		output_2 = tf.reshape(input_2, [-1, 1, input_2.get_shape().as_list()[1], 1])
		with tf.variable_scope('conv_1'):
			output_2 = conv_bn_relu_dropout(output_2, 
				[1, 260, 1, 4], 10, keep_prob, is_training)
			output_2 = max_pool_1d(output_2, 10)
		with tf.variable_scope('conv_2'):
			output_2 = conv_bn_relu_dropout(output_2, 
				[1, 200, 4, 8], 10, keep_prob, is_training)
			output_2 = max_pool_1d(output_2, 10)

	output = tf.concat([tf.reshape(output_2, 
		[-1, np.prod(output_2.get_shape().as_list()[1:])]), output_1], 1)
	with tf.variable_scope('second_fc'):
		output = output_layer(output, 1000)
		output = tf.nn.relu(output)
		output = tf.nn.dropout(output, keep_prob)
	with tf.variable_scope('third_fc'):
		output = output_layer(output, 50)
		output = tf.nn.relu(output)
		output = tf.nn.dropout(output, keep_prob)
	# with tf.variable_scope('forth_fc'):
	# 	output = output_layer(output, 100)
	# 	output = tf.nn.relu(output)
	with tf.variable_scope('output'):
		output = tf.nn.sigmoid(output_layer(output, output_dim))
	return output


def densenet(input, num_labels, depth, growth, keep_prob, is_training):
	'''
	The densenet
	:param input: A tensor, the original input
	:num_labels: A scalar, the number of candidate labels
	:depth: A scalar, the total number of layers in the whole network: 40, 100...
	:growth: A scalar, the growth rate: 12, 16, 32...
	:keep_prob: A scalar, for the dropout
	:is_training: A bool, for the batch norm
	'''
	layers = (depth - 4) / 3

	current = conv2d(input, input.get_shape().as_list()[-1], 16, 3)
	with tf.variable_scope('block_1'):
		current, features = densenet_block(current, layers, 16, growth, keep_prob, is_training)
		current = bn_relu_conv_dropout(current, features, features, 1, is_training, keep_prob=keep_prob)
		current = avg_pool(current, 2)
	with tf.variable_scope('block_2'):
		current, features = densenet_block(current, layers, features, growth, keep_prob, is_training)
		current = bn_relu_conv_dropout(current, features, features, 1, is_training, keep_prob=keep_prob)
		current = avg_pool(current, 2)
	with tf.variable_scope('block_3'):
		current, features = densenet_block(current, layers, features, growth, keep_prob, is_training)
		current = batch_normalization_layer(current, is_training)
		current = tf.nn.relu(current)
		current = avg_pool(current, 8)

	final_dim = np.prod(current.get_shape().as_list()[1:])
	x_trans = tf.reshape(current, [ -1, final_dim ])
	with tf.variable_scope('last_layer'):
		y_logit = output_layer(x_trans, num_labels)

	return y_logit, x_trans

def resnet(input_tensor_batch, num_class, n, keep_prob, is_training, SE=False, ratio=16):
	'''
	The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
	:param input_tensor_batch: 4D tensor
	:param num_class: the output number of classes
	:param n: num_residual_blocks
	:param keep_prob: for dropout
	:is_training: for batch norm
	:SE: bool, if we use the senet
	:ratio: the se ratio
	:return: last layer in the network. Not softmax-ed
	'''
	layers = []
	with tf.variable_scope('conv0'):
		conv0 = conv_bn_relu(input_tensor_batch, 
			[3, 3, input_tensor_batch.get_shape().as_list()[-1], 16], 1, is_training)
		layers.append(conv0)

	for i in range(n):
		with tf.variable_scope('conv1_%d' %i):
			if i == 0:
				conv1 = residual_block(layers[-1], 16, is_training, keep_prob=keep_prob,
					first_block=True, SE=SE, ratio=ratio)
			else:
				conv1 = residual_block(layers[-1], 16, is_training, keep_prob=keep_prob,
					SE=SE, ratio=ratio)
			layers.append(conv1)

	for i in range(n):
		with tf.variable_scope('conv2_%d' %i):
			conv2 = residual_block(layers[-1], 32, is_training, keep_prob=keep_prob,
				SE=SE, ratio=ratio)
			layers.append(conv2)

	for i in range(n):
		with tf.variable_scope('conv3_%d' %i):
			conv3 = residual_block(layers[-1], 64, is_training, keep_prob=keep_prob,
				SE=SE, ratio=ratio)
			layers.append(conv3)
		# only used for 32*32*3 inputs
		# assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

	with tf.variable_scope('fc'):
		in_channel = layers[-1].get_shape().as_list()[-1]
		bn_layer = batch_normalization_layer(layers[-1], is_training)
		relu_layer = tf.nn.relu(bn_layer)
		global_pool = tf.reduce_mean(relu_layer, [1, 2])

		assert global_pool.get_shape().as_list()[-1:] == [64]
		with tf.variable_scope('last_layer'):
			output = output_layer(global_pool, num_class)
		layers.append(output)

	return layers[-1], global_pool


def se_resnet(input_tensor_batch, num_class, n, keep_prob, is_training, ratio=16):
	return resnet(input_tensor_batch, num_class, n, keep_prob, is_training, SE=True, ratio=ratio)

def simple_conv(inputs, num_class, keep_prob, is_training):
	internal_layers = list()
	with tf.variable_scope('conv_1'):
		current = conv_bn_relu_dropout(inputs, 
			[3, 3, inputs.get_shape().as_list()[-1], 32], 1, keep_prob, is_training)
		internal_layers.append(current)
		current = avg_pool(current, 2)
	with tf.variable_scope('conv_2'):
		current = conv_bn_relu_dropout(current, [3, 3, 32, 64], 1, keep_prob, is_training)
		internal_layers.append(current)
		current = avg_pool(current, 2)
	with tf.variable_scope('conv_3'):
		current = conv_bn_relu_dropout(current, [3, 3, 64, 128], 1, keep_prob, is_training)
		internal_layers.append(current)
		current = avg_pool(current, 2)
	with tf.variable_scope('conv_4'):
		current = conv_bn_relu_dropout(current, [3, 3, 128, 64], 1, keep_prob, is_training)
		internal_layers.append(current)
		current = avg_pool(current, 2)
	with tf.variable_scope('conv_5'):
		current = conv_bn_relu_dropout(current, [1, 1, 64, 32], 1, keep_prob, is_training)
		internal_layers.append(current)
		current = avg_pool(current, 2)
	final_dim = np.prod(current.get_shape().as_list()[1:])
	x_trans = tf.reshape(current, [ -1, final_dim ])
	internal_layers.append(x_trans)
	with tf.variable_scope('last_layer'):
		output = output_layer(x_trans, num_class)
	return output, x_trans

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





