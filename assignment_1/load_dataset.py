# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:43:17 2017

@author: xliu
"""

import os
import pickle
import numpy as np

def loadData(root):
	xs = []
	ys = []
	
	for b in range(1, 6):
		file = os.path.join(root, 'data_batch_%d' % (b, ))
		with open(file, 'rb') as f:
			rawDataDict = pickle.load(f, encoding = 'latin1')
			X = rawDataDict['data']
			Y = rawDataDict['labels']
			X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
			Y = np.array(Y)
			# append to subset
			xs.append(X)
			ys.append(Y)
	
	# concatenate all subset X and Y
	Xtrain = np.concatenate(xs)
	Ytrain = np.concatenate(ys)
	del X, Y
	
	# read-in the test dataset
	testFile = os.path.join(root, 'test_batch')
	with open(testFile, 'rb') as f:
		rawDataDict = pickle.load(f, encoding = 'latin1')
		Xtest = rawDataDict['data']
		Ytest = rawDataDict['labels']
		Xtest = Xtest.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
		Ytest = np.array(Ytest)
			
	
	return Xtrain, Xtest, Ytrain, Ytest
	
	
def rel_error(x, y):
	'''return the relative error'''
	error = np.max((np.abs(x - y)) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
	return error
	
if __name__ == '__main__':
	root = os.path.join('/Users/xliu/Documents/MRC/Work/Online course/',
    'CS231N Convolutional Neural Networks for Visual Recognition/',
	'spring1617_assignment1/assignment1/cs231n/datasets/cifar-10-batches-py')
	print(root)
	Xtrain, Xtest, Ytrain, Ytest = loadData(root)
	print(Xtrain.shape)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	