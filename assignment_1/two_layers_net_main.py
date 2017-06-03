# -*- coding: utf-8 -*-
"""
This is the main function to train two layers fully 
connected neural networks.

@author: xliu
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from load_dataset import *

import neural_network_class
from importlib import reload
reload(neural_network_class)
from neural_network_class import *

# define the root of data files
root = os.path.join('/Users/xliu/Documents/MRC/Work/Online course/',
    'CS231N Convolutional Neural Networks for Visual Recognition/',
'assignment1/assignment1/cs231n/datasets/cifar-10-batches-py')

# read-in the data and generate training and test sets from external module
#Xtrain, Xtest, ytrain, ytest = loadData(root)
Xtrain, Xtest, ytrain, ytest = loadData(root)

# print-out shape of training/test dataset as sanity check
print('Training data shape', Xtrain.shape)
print('Training lavel shape', ytrain.shape)
print('Test data shape', Xtest.shape)
print('Test label shape', ytest.shape)


# plot some images and visualize how they look like
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
			'ship', 'truck']
num_classes = len(classes)

sample_per_class = 7

#for y, cls in enumerate(classes):
#	idxs = np.flatnonzero(ytrain == y)
#	idxs = np.random.choice(idxs, sample_per_class, replace = False)
#	for i, idx in enumerate(idxs):
#		plt_idx = i * num_classes + y + 1
#		plt.subplot(sample_per_class, num_classes, plt_idx)
#		plt.imshow(Xtrain[idx].astype('uint8'))
#		plt.axis('off')
#		if i == 0:
#			plt.title(cls)

# split the training set into sub-training, validation and dev
# where dev is the very small size dataset from sub-training,
# it is used for developing function and perform sanity check
number_training = 49000
number_validate = 1000
number_test = 1000
number_dev = 500

mask = range(number_training)
X_train = Xtrain[mask]
y_train = ytrain[mask]

mask = range(number_training, number_training + number_validate)
X_valid = Xtrain[mask]
y_valid = ytrain[mask]

mask = np.random.choice(number_training, number_dev, replace = False)
X_dev = Xtrain[mask]
y_dev = ytrain[mask]

mask = range(number_test)
X_test = Xtest[mask]
y_test = ytest[mask]

# all data need to be subtracted by the samples' mean values
# before we can calculate the mean values, we need to convert
# the dataset from 4-D into 2-D. The reason of converting
# raw dataset from 2-D to 4-D is due to we want to plot the image
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_valid = np.reshape(X_valid, (X_valid.shape[0], -1))
X_dev   = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test  = np.reshape(X_test, (X_test.shape[0], -1))

cases_mean_values = np.mean(X_train, axis = 0)

# plot the image of mean value
#plt.imshow(cases_mean_values.reshape((32, 32, 3)).astype('uint8'))

# substract mean values in X feature space
X_train -= cases_mean_values
X_test -= cases_mean_values
X_dev -= cases_mean_values
X_valid -= cases_mean_values

# augment X space with a padding column
#X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
#X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
#X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
#X_valid = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])

#################################################################
#
#
#
#	two layers neural netwrok classifier section
#
#
#
#################################################################

'''

before construct the actual network, we start from 
performing a simplest toy example, developed codes are in neural_network_develop.py

'''
from neural_network_class import *

intput_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10

nn_clf = neural_network(intput_size, hidden_size, num_classes)

kwargs = {'reg':0.25, 'learn_rate':1e-4, 'learn_rate_decay':0.95,
	       'numIters': 1000, 'batch_size':200, 'verbose':True}

# train the network
train_hist = nn_clf.train(X_train, y_train, X_valid, y_valid, **kwargs)

# validation accuracy
val_acc = np.mean(nn_clf.predict(X_valid) == y_valid)
print('validation accuracy: ', val_acc)


# debug the training
#plt.subplot(2,1,1)
#plt.plot(train_hist['loss_history'])
#plt.title('loss history')
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#
#plt.subplot(2,1,2)
#plt.plot(1-np.array(train_hist['train_accuracy']), label='train', color='red', alpha=0.8)
#plt.plot(1-np.array(train_hist['validation_accuracy']), label='validation', color='blue', alpha=0.8)
#plt.title('classification error history')
#plt.xlabel('Epoch')
#plt.ylabel('Classification error')
#plt.legend(loc='best')
#plt.show()
''' conclusion from learning curve: the current nerual network is underfitting'''

#from vis_utils import visualize_grid
#
#def show_net_weights(nn_class):
#	W1 = nn_class.params['W1']
#	W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
#	plt.imshow(visualize_grid(W1, padding = 3).astype('uint8'))
#	plt.gca().axis('off')
#	plt.show()
#
#show_net_weights(nn_clf)



#################################################################
#
#
#
#	fine tune two layers neural network
#
#
#
#################################################################

'''
Looking at the visualizations above, we see that the loss is decreasing more 
or less linearly, which seems to suggest that the learning rate may be too low. 
Moreover, there is no gap between the training and validation accuracy, 
suggesting that the model we used has low capacity, 
and that we should increase its size. On the other hand, 
with a very large model we would expect to see more overfitting, 
which would manifest itself as a very large gap between the training and validation 
accuracy.
'''

#################################################################################
# Tune hyperparameters using the validation set. Store your best trained        #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################

from itertools import product

# set candidate hyper-parameters
iterNum = [5000, 20000]

# solve numerical issue
learn_rate = [1e-3, 3e-3]
#learn_rate = np.logspace(-10, 0, 3)

reg = [0.4, 1]
#reg = np.logspace(-3, 5, 3)

hidden_size = [300, 500]

hypara_comb = list(product(iterNum, learn_rate, reg, hidden_size))

best_val_accuracy = -1
best_hyparams     = None
best_nn           = None
train_hist_cv ={}
for n, hypara in enumerate(hypara_comb):
	nn_cls_cv = neural_network(intput_size, hypara[3], num_classes)
	kwargs = {'reg':hypara[2], 'learn_rate':hypara[1], 'learn_rate_decay':0.95,
	       'numIters': hypara[0], 'batch_size':200, 'verbose':True}
	
	# train the network
	train_hist = nn_cls_cv.train(X_train, y_train, X_valid, y_valid, **kwargs)
	train_hist_cv[n] = train_hist
	# validation accuracy
	val_acc = np.mean(nn_clf.predict(X_valid) == y_valid)
	print('validation accuracy: ', val_acc)
	
	if val_acc > best_val_accuracy:
		best_val_accuracy = val_acc
		best_nn = nn_cls_cv
		best_hyparams = kwargs





























