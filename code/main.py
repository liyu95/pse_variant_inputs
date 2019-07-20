#!/usr/bin/env python
from parameter import *
import tensorflow as tf
from model import simple_conv, fc_net, attention
from utils import *
from tf_utils import *
from tf_utils import config
import sys
from scipy.stats import pearsonr
import pdb

def train_model(X, Y, testX, testY):

	# create placeholder
	num_class = Y.shape[-1]
	xs_shape = [None]+list(X.shape)[1:]
	xs = tf.placeholder("float", shape=xs_shape)
	ys = tf.placeholder("float", shape=[None, num_class])
	lr = tf.placeholder("float", shape=[])
	train_flag = tf.placeholder("bool", shape=[])
	if model_name=='attention':
		mask = tf.placeholder("float", shape=[None, 1, 1, X.shape[1]])
	keep_prob = tf.placeholder(tf.float32)

	# graph output
	if model_name=='simple_conv':
		y_logit, x_trans = simple_conv(xs)
	elif model_name=='fc':
		y_logit, x_trans = fc_net(xs, num_class, keep_prob, train_flag)
	elif model_name=='attention':
		y_logit, x_trans = attention(xs, train_flag, mask)
	else:
		sys.exit("Network not defined!")

	# define accuracy
	# correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(ys, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# tf.summary.scalar('accuracy',accuracy)

	# define loss
	mse = tf.reduce_mean((tf.losses.mean_squared_error(
	    predictions=y_logit, labels=ys)))
	tf.summary.scalar('mse',mse)
	regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.add_n([mse] + regu_losses)

	# define optimizer
	#train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)
	train_step = tf.train.AdamOptimizer(lr).minimize(loss)

	# define session
	saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	session = tf.Session(config=config)
	session.run(tf.global_variables_initializer())

	if from_ckpt and tf.train.checkpoint_exists:
		saver.restore(session, ckpt_path)
		print('Restored from checkpoint...')

	merged=tf.summary.merge_all()
	train_writer=tf.summary.FileWriter(run_dir+'/train_log',session.graph)
	test_writer=tf.summary.FileWriter(run_dir+'/test_log')

	train_data = X
	train_labels = Y 
	batch_count = len(train_data) / batch_size
	batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
	batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
	learning_rate = init_lr
	print("Start training, batch per epoch: ", batch_count)
	step = 0
	for epoch in xrange(1, 1+epoches):
		if epoch in lr_dict:
			learning_rate = lr_dict[epoch]
		for batch_idx in xrange(batch_count):
			step = step +1
			xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
			feed_dict = { xs: xs_, ys: ys_, lr: learning_rate, 
			  				train_flag: True, keep_prob: keep_ratio}
			if model_name=='attention':
				feed_dict.update({mask:create_padding_mask(np.sum(xs_, -1))})
			batch_res = session.run([merged, train_step, mse ],
				feed_dict = feed_dict)
			train_writer.add_summary(batch_res[0], step)
			if batch_idx % output_step == 0:
				print(epoch, batch_idx, batch_res[2:])
				test_idx = np.random.choice(len(testX), test_batch)
				feed_dict = { xs: testX[test_idx], 
					ys: testY[test_idx], lr: learning_rate,
					train_flag: False, keep_prob: 1}
				if model_name=='attention':
					feed_dict.update({mask:create_padding_mask(np.sum(xs_, -1))})
				test_summary = session.run([merged, ys, y_logit], 
					feed_dict = feed_dict)
				test_writer.add_summary(test_summary[0], step)
				# print(test_summary[1])
				# print(test_summary[2])
				print('PearsonR:')
				print(pearsonr(test_summary[1][:,0], 
					test_summary[2][:, 0]))

		save_path = saver.save(session, ckpt_path)
		# pdb.set_trace()
		if model_name=='attention':
			test_results = run_in_batch_avg(session, [mse ], [ xs, ys, mask ],
				feed_dict = { xs: testX, ys: testY, train_flag: False, keep_prob: 1.,
					mask: create_padding_mask(np.sum(testX, -1))}, 
				batch_size = test_batch)
		else:
			test_results = run_in_batch_avg(session, [mse ], [ xs, ys ],
				feed_dict = { xs: testX, ys: testY, train_flag: False, keep_prob: 1.}, 
				batch_size = test_batch)			
		print(epoch, batch_res[2:], test_results)



if __name__ == '__main__':
	set_up_log_dir()
	# print_parameters()
	# Remember to normalize the input data
	# X, Y, testX, testY = prepare_data_cifar10()
	if model_name=='fc':
		X, Y, testX, testY = load_data_eagle()
	else:
		X, Y, testX, testY = load_data_eagle_for_attention()
	train_model(X, Y, testX, testY)
