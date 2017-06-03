# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:08:46 2017

This is the module for building softmax classifier.
Codes are implementing loss and gradient evaluation.

Input:

	W: D * C; where D is the number of features and C is the number of classes
	X: N * D; where N is the number of data points and D is the number of features
	y: N * 1
	reg: scalar

Output:
	
	loss: scalar quantifies the current error between prediction made by
	      the underlying weight matrix.
	dw: D * C; where it is the gradient of loss function wrt weights
	
@author: xliu
"""

import numpy as np
from gradient_check import gradient_check

def softmax_crossEntropy_naive(W, X, y, reg):
	# 1.1 retrieve dimension information
	num_train = X.shape[0]
	num_feat  = X.shape[1]
	num_class = len(np.unique(y))
	
	# 1.2 initialize outputs 
	dw = np.zeros_like(W)
	loss = 0
	
	# 2.1 loop-over all data points in data set and evaluate the loss
	for n in range(num_train):
		
		# 2.1.1 evaluate the score by dot product feature and weight
		score = X[n].dot(W)
		
		# 2.1.2 substract score by its maximum value, due to the exponentiate
		# a very large number will cause numerical instability
		score -= np.max(score)
		
		# 2.1.3 evaluate the normalized 'probability' for each class
		prob_cls = np.exp(score[y[n]]) / np.sum(np.exp(score))
		
		loss -= np.log(prob_cls)
		
		# 3.1 evluate the gradient of loss wrt weights for every features
		# it has two loops, 
		# first -- number of data points
		#	second -- number of classes
		for c in range(num_class):
			if c == y[n]:
				dw[:, c] -= (X[n, c] -  prob_cls * X[n, c])
			else:
				#dw[:, c] = 0
				dw[:, c] = X[n, c] * (np.exp(score[c]) / np.sum(np.exp(score)))
				
	# 3.1 sum of losses divided by number of data points and plus regularization
	loss = loss/float(num_train) + 0.5 * reg * np.sum(W * W)
	
	# 3.2 evluate the gradient of loss wrt weights for every features
	dw /= float(num_train)
	dw += reg * W
	
	
	# return results
	return loss, dw


#########################################################
#
#
#	vectorized softmax
#
#
#########################################################
def softmax_crossEntropy_vectorized(W, X, y, reg):
	# 1. initialize outputs
	loss = 0
	dw = np.zeros_like(W)
	num_train = X.shape[0]
	# 2.1 evaluate the score by dot product W and X
	score = X.dot(W)
	
	# 2.2 for sake of numerical stability, we substract the maximum value
	# in each row
	score -= np.max(score, axis = 1, keepdims = True)
	
	# 2.3 normalized probabilities
	score_prob_sum = np.sum(np.exp(score), axis = 1, keepdims = True)
	score_prob_normalized = np.exp(score) / score_prob_sum
	#prob_normalized = np.exp(score[np.arange(num_train), y]).reshape(-1,1)/np.sum(np.exp(score), axis=1, keepdims = True)
	
	# 2.4 evaluate loss
	#loss -= np.sum(np.log(prob_normalized))
	loss = np.sum(-np.log(score_prob_normalized[np.arange(num_train), y]))
	loss = loss / float(num_train) + 0.5 * reg * np.sum(W * W)
	
	# 3.1 binary indicator to imply i-th is the true label location
	indicator = np.zeros_like(score_prob_normalized)
	indicator[np.arange(num_train), y] = 1
	dw = X.T.dot(score_prob_normalized - indicator)
	dw = dw / float(num_train)
	dw += reg * W
	
	return loss, dw






if __name__ == '__main__':
	weights = 0.0000001*np.random.randn(X_dev.shape[1], len(np.unique(y_dev)))
	loss, grad = softmax_crossEntropy_vectorized(weights, X_dev, y_dev, 0.0)
	print(loss)
	#f = lambda w: softmax_crossEntropy_naive(w, X_dev, y_dev, 0.0)[0]
	#grad_numerical = gradient_check(f, weights, grad)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	