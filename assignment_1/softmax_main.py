# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:41:44 2017

@author: xliu
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from load_dataset import *

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

for y, cls in enumerate(classes):
	idxs = np.flatnonzero(ytrain == y)
	idxs = np.random.choice(idxs, sample_per_class, replace = False)
	for i, idx in enumerate(idxs):
		plt_idx = i * num_classes + y + 1
		plt.subplot(sample_per_class, num_classes, plt_idx)
		plt.imshow(Xtrain[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(cls)

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
plt.imshow(cases_mean_values.reshape((32, 32, 3)).astype('uint8'))

# substract mean values in X feature space
X_train -= cases_mean_values
X_test -= cases_mean_values
X_dev -= cases_mean_values
X_valid -= cases_mean_values

# augment X space with a padding column
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
X_valid = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])
#################################################################
#
#
#
#	Softmax classifier section
#
#
#
#################################################################

from gradient_check import gradient_check
from softmax_fcn import *
import time

weights = 0.0001*np.random.randn(X_dev.shape[1], len(np.unique(y_dev)))
loss, grad = softmax_crossEntropy_naive(weights, X_dev, y_dev, 0.0)
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

f = lambda w: softmax_crossEntropy_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = gradient_check(f, weights, grad, 10)

# do another sanity check with regualarization
loss, grad = softmax_crossEntropy_naive(weights, X_dev, y_dev, 1e2)
f = lambda w: softmax_crossEntropy_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = gradient_check(f, weights, grad, 10)

# time the computational cost of naive gradient/loss evaluation
tic = time.time()
loss_naive, grad_naive = softmax_crossEntropy_naive(weights, X_dev, y_dev, 1e-4)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc-tic))

# sanity check of vectorized gradient/loss evaluation
f = lambda w: softmax_crossEntropy_vectorized(w, X_dev, y_dev, 0.0)[0]
grad_numerical = gradient_check(f, weights, grad, 10)

# time the computational difference between naive and vectorized
tic = time.time()
loss_vectorized, grad_vectorize = softmax_crossEntropy_vectorized(weights, X_dev, y_dev, 1e-4)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc-tic))

#################################################################
#
#
#
#	Softmax classifier class to train and predict
#
#
#
#################################################################

















