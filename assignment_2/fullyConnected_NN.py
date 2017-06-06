# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:49:29 2017

@author: xliu
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from loadData import *


import class_fullyConnectedNet
from importlib import reload
reload(class_fullyConnectedNet)

from class_fullyConnectedNet import * 

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load-in data
# define the root of data files
cifar10_dir = os.path.join('/Users/xliu/Documents/MRC/Work/Online course/',
    'CS231N Convolutional Neural Networks for Visual Recognition/',
'assignment1/assignment1/cs231n/datasets/cifar-10-batches-py')

read_data = False

if read_data == True:
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	
	# As a sanity check, we print out the size of the training and test data.
	print('Training data shape: ', X_train.shape)
	print('Training labels shape: ', y_train.shape)
	print('Test data shape: ', X_test.shape)
	print('Test labels shape: ', y_test.shape)
	
	
	data = subsampleData(X_train, y_train, X_test, y_test)
	for k, v in data.items():
	    print('%s: ' % k, v.shape)

print('data has not been read.')
############################



# Use a three-layer Net to overfit 50 training examples.

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

#weight_scale = 1e-2
#learning_rate = 1e-2
#model = fullyConnectedNet([100, 100], input_dims = 3 * 32 * 32, num_classes = 10,
#              weight_scale=weight_scale, dtype=np.float64)
#Solver1 = solver(model, small_data,
#                print_every=10, num_epochs=20, batch_size=25,
#                update_rule='sgd',
#                optim_config={
#                  'learning_rate': learning_rate,
#                }
#         )
#Solver1.train()
#
#plt.subplot(2, 1, 1)
#plt.plot(range(len(Solver1.loss_history)), Solver1.loss_history)
#plt.title('Training loss history')
#plt.xlabel('Iteration')
#plt.ylabel('Training loss')
#
#plt.subplot(2, 1, 2)
#plt.title('Error')
#plt.plot(Solver1.train_err_history, '-o', label='train')
#plt.plot(Solver1.valid_err_history, '-o', label='val')
#plt.plot([0.5] * len(Solver1.valid_err_history), 'k--')
#plt.xlabel('Epoch')
#plt.legend(loc='lower right')
#plt.gcf().set_size_inches(15, 12)
#plt.show()
#
#
## Use a five-layer Net to overfit 50 training examples.
#
#num_train = 50
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#learning_rate = 1e-2
#weight_scale = 1e-5
#model = fullyConnectedNet([100, 100, 100, 100], input_dims = 3 * 32 * 32, num_classes = 10,
#                weight_scale=weight_scale, dtype=np.float64)
#Solver2 = solver(model, small_data,
#                print_every=10, num_epochs=20, batch_size=25,
#                update_rule='sgd',
#                optim_config={
#                  'learning_rate': learning_rate,
#                }
#         )
#Solver2.train()
#
#plt.subplot(2, 1, 1)
#plt.plot(range(len(Solver2.loss_history)), Solver1.loss_history)
#plt.title('Training loss history')
#plt.xlabel('Iteration')
#plt.ylabel('Training loss')
#
#plt.subplot(2, 1, 2)
#plt.title('Error')
#plt.plot(Solver2.train_err_history, '-o', label='train')
#plt.plot(Solver2.valid_err_history, '-o', label='val')
#plt.plot([0.5] * len(Solver1.valid_err_history), 'k--')
#plt.xlabel('Epoch')
#plt.legend(loc='lower right')
#plt.gcf().set_size_inches(15, 12)
#plt.show()



# ---------------------------------------------------------------------
#
#
#
#        randomly chosen weight_scale and learning_rate test

not_reach = True

while not_reach:
	weight_scale = 10**(np.random.uniform(-6,-1))
	learning_rate = 10**(np.random.uniform(-4,-1))
	loss_history, train_err_hist = run_model(small_data, hidden_dims,input_dims, num_classes, weight_scale,learning_rate)
	if min(train_err_hist) == 0.0:
		not_reach = False
		overfit_loss = loss_history
		lr = learning_rate
		ws = weight_scale

print('Has worked with %f and %f'%(lr,ws))
plt.plot(overfit_loss, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

loss_history, train_err_hist = run_model(small_data, hidden_dims,\
								  input_dims, num_classes, ws, lr)
print('Has worked with %f and %f'%(lr,ws))
plt.plot(loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

#
#
#
# end of randomly chosen weight_scale and learning_rate test
# ---------------------------------------------------------------------














































