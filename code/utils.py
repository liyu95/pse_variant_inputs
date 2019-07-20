#!/usr/bin/env python
from itertools import cycle
import numpy as np
from parameter import *
from tflearn.datasets import cifar10
from tflearn.data_utils import to_categorical
from collections import Counter
import os
import pandas as pd
from tf_utils import get_available_gpus
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from sklearn import preprocessing
import pdb

def print_parameters():
	print('='*40)
	print('Device information: '+str(get_available_gpus()))
	print('Here is the hyper parameter summary:')
	print('Model type: '+model_name)
	if model_name=='densenet':
		print('Layers: '+str(layers))
		print('Growth: '+str(growth))
	if model_name=='resnet':
		print('Blocks: '+str(blocks))
	if model_name=='se_resnet':
		print('Blocks: '+str(blocks))
		print('SE_ratio: '+str(se_ratio))
	print('Keep ratio: '+str(keep_ratio))
	print('Total epoch: '+str(epoches))
	print('Initial learning rate: '+str(init_lr))
	print('Learning rate decay: '+str(lr_dict))
	print('Weight decay: '+str(weight_decay))
	print('Log folder: '+run_dir)
	print('='*40)

class batch_object(object):
	"""docstring for batch_object"""
	def __init__(self, data_list, batch_size):
		super(batch_object, self).__init__()
		self.data_list = data_list
		self.batch_size = batch_size
		self.pool = cycle(data_list)
		
	def next_batch(self):
		data_batch = list()
		for i in xrange(self.batch_size):
			data_batch.append(next(self.pool))
		data_batch = np.array(data_batch)
		return data_batch

def set_up_log_dir():
	'''
	check if the run dir exist of not
	if so, overwrite, if not, create it
	'''
	if not os.path.isdir(run_dir):
		os.system('mkdir '+run_dir)
		os.system('mkdir '+run_dir+'train_log')
		os.system('mkdir '+run_dir+'test_log')
	else:
		os.system('rm '+run_dir+'train_log/*')
		os.system('rm '+run_dir+'test_log/*')


# prepare the data, take the cifar10 as an example
def prepare_data_cifar10():
	'''
	data preparation function
	:param X: 4d array, samples*H*W*channels, training data
	:param Y: 2d array, samples*num_class
	:param testX: 4d array, samples*H*W*channels, testing data
	:param testY: 2d array, samples*num_class
	'''
	(X, Y), (testX, testY) = cifar10.load_data()
	num_class = len(Counter(Y))
	Y = to_categorical(Y, num_class)
	testY = to_categorical(testY, num_class)
	return X, Y, testX, testY


def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):
	'''
	whole set performance check on the testing data
	'''                            
	res = [ 0 ] * len(tensors)                                                                                           
	batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
	total_size = len(batch_tensors[0][1])                                                                                
	batch_count = (total_size + batch_size - 1) / batch_size                                                             
	for batch_idx in xrange(batch_count):                                                                                
		current_batch_size = None                                                                                          
		for (placeholder, tensor) in batch_tensors:                                                                        
			batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
			current_batch_size = len(batch_tensor)                                                                           
			feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
		# pdb.set_trace()
		tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
		res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
	return [ r / float(total_size) for r in res ]



def create_padding_mask(seq):
    seq = np.equal(seq, 0).astype(int)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


def load_data_eagle():
	eagle_1_path = '../data/eagle_1.csv'
	eagle_2_path = '../data/eagle_2.csv'
	eagle_1 = pd.read_csv(eagle_1_path)
	eagle_2 = pd.read_csv(eagle_2_path)
	X = eagle_2.iloc[:,:58].values
	Y = eagle_2.iloc[:,58:].values
	testX = eagle_1.iloc[:,:58].values
	testY = eagle_1.iloc[:,58:].values
	
	if less2more:
		X = eagle_1.iloc[:,:58].values
		Y = eagle_1.iloc[:,58:].values
		testX = eagle_2.iloc[:,:58].values
		testY = eagle_2.iloc[:,58:].values

	return X,Y,testX,testY

def load_data_eagle_for_attention():
	eagle_1_path = '../data/eagle_1.csv'
	eagle_2_path = '../data/eagle_2.csv'
	eagle_1 = pd.read_csv(eagle_1_path)
	eagle_2 = pd.read_csv(eagle_2_path)
	X = eagle_2.iloc[:,:58].values
	Y = eagle_2.iloc[:,58:].values
	testX = eagle_1.iloc[:,:58].values
	testY = eagle_1.iloc[:,58:].values

	# converting X to [n, 14, 5]
	# converting Y to [n, 15, 2]
	X_temp = X[:, :56].reshape([-1, 14, 4])
	X_to_pad = X[:, 56:58]
	X_to_pad = np.tile(X_to_pad, (1,14)).reshape([-1, 14, 2])
	X = np.concatenate([X_temp, X_to_pad], axis=-1)
	
	testX_temp = testX[:, :56].reshape([-1, 14, 4])
	testX_to_pad = testX[:, 56:58]
	testX_to_pad = np.tile(testX_to_pad, (1,14)).reshape([-1, 14, 2])
	testX = np.concatenate([testX_temp, testX_to_pad], axis=-1)

	# Y = Y.reshape([-1, 2, 14])
	# Y = np.swapaxes(Y, 1, 2)
	# testY = testY.reshape([-1, 2, 14])
	# testY = np.swapaxes(testY, 1, 2)

	
	if less2more:
		X, testX = testX, X
		Y, testY = testY, Y
		
	return X,Y,testX,testY

