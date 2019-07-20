#!/usr/bin/env python

import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--id', help='the run id to identity the run.', required=True)
# parser.add_argument('--model', help='the model type, resnet or densenet', required=True)
# parser.add_argument('--layers', help='the number of layers for densenet', type=int)
# parser.add_argument('--growth', help='the growth rate for densenet', type=int)
# parser.add_argument('--blocks', help='the number of blocks for resnet', type=int)
# args = parser.parse_args()

# The GPU resource and the warning off
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
parameters related to the log
run_dir: all the log and the saved model are in run_dir
	different programs can run at the same time with 
	different run_id
output_step: The frequency of output the training acc 
	on the screen
'''
# run_id = args.id
run_id = 'att_14_to_8'
run_dir = run_id + '/'
output_step = 10

'''
parameters related to the model configuration
candidate model: densenet, resnet, senet, simple_conv
weight_decay: weight regularizer coefficient
params related to densenet:
	layers: 40, 100, ...
	growth: 12, 16, 32, ...
params related to resnet:
	blocks: 5 (32), 9 (56), 18 (110), 27(164)
params related to se_resnet:
	blocks: the same as the resnet
	se_ratio: the squeeze ratio
'''
# model_name = args.model
model_name = 'attention'
weight_decay = 0.0002


'''
parameters relate to the training process
batch_size: training batch size
test_batch: testing batch size
epoches: the total training epoches
init_lr: the initial learning rate
lr_dict: define the learning rate decay
keep ratio: the dropout keep ratio
full check: whether check the performance on the whole 
	test set after each epoch
MCF_REG: whether use the MCF regularizer or not, it can be slow and 
	take a lot of memory
'''
batch_size = 8192
test_batch = 8192
epoches = 200
init_lr = 0.01
# optimized for densenet and resnet
lr_dict = {10: 0.001, 20:0.0001}
keep_ratio = 0.8
full_check = True
less2more = False


'''
model persistent
from_ckpt: restore the model and further training
ckpt_path: define the saved model name
'''
from_ckpt = False
ckpt_path = run_dir+model_name+'.ckpt'
