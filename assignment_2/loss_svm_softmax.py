# -*- coding: utf-8 -*-
"""
svm loss function and softmax loss function

@author: xliu
"""

import numpy as np
from gradient_check import *


def svm_loss(x, y):
    num_train = x.shape[0]
    
    margins = np.maximum(0, x-x[np.arange(num_train), y].reshape(-1,1) + 1)
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins)/num_train
    
    num_pos_margin = np.sum(margins > 0, axis = 1)
    
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(num_train), y] -= num_pos_margin
    dx /= num_train
    
    return loss, dx
				
def softmax_loss(x, y):
    '''
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    '''
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx				






if __name__ == '__main__':
	num_classes, num_inputs = 10, 50
	x = 0.001 * np.random.randn(num_inputs, num_classes)
	y = np.random.randint(num_classes, size=num_inputs)
	
	# sanity check of svm loss function
	dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
	loss, dx = svm_loss(x, y)
	# Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
	print('Testing svm_loss:')
	print('loss: ', loss)
	print('dx error: ', rel_error(dx_num, dx))

	# sanity check of softmax loss function
	dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
	loss, dx = softmax_loss(x, y)
	
	# Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
	print('\nTesting softmax_loss:')
	print('loss: ', loss)
	print('dx error: ', rel_error(dx_num, dx))
































