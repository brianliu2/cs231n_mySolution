# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:22:02 2017

@author: xliu
"""
import numpy as np
from gradient_check import *

def rel_error(x, y):
    '''returns relative error'''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def affine_forward(X, W, b):
    ##############################################################
    #                                                            #
    # This is actually a unit for doing matrix multiplication.   #
    # out can be thinking as the output of this layer            #
    # cache can be thinking as the input of this layer           #
    #                                                            #
    ##############################################################
    '''
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension C.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, C)
    - b: A numpy array of biases, of shape (C,)

    Returns a tuple of:
    - out: output, of shape (N, C)
    - cache: (x, w, b)
    '''
    # 1. we need to 'roll' input space into a matrix form
    X = X.reshape(X.shape[0], W.shape[0])
    
    # 2. do forward action by multiplying input and weight
    output = X.dot(W) + b
    cache = (X, W, b)
    
    # 3. return outputs and intermediate information
    return output, cache



def affine_backward(dout, cache):
    ##############################################################
    #                                                            #
    # This is actually doing back propagation.                   #
    # dout is the derivative of loss wrt outputs of layer ahead  #
    # cache is the input of current layer, and it is stored by   #
    # affine_forward function.                                   #
    #                                                            #
    ##############################################################
    '''
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: derivative of loss wrt to unit of one layer ahead, of shape (N, C)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, C)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, C)
    - db: Gradient with respect to b, of shape (C,)
    '''
    
    # 1. retrieve the data externally
    X, W, b = cache
    dx, dw, db = None, None, None
    
    # 2. given the derivative of loss wrt to unit of one layer ahead, 
    # dl/dx = delta^{L+1} * df/dx; where f = w*x+b
    dx = dout.dot(W.T).reshape(X.shape)
    #dx = np.dot(W, dout.T)
    #dx = np.dot(dout.T, W)
    #dx = np.dot(dout.T, W.T)
    #dx = dout.dot(W)
    
    # 3. given the derivative of loss wrt to unit of one layer ahead, 
    # dl/dw = delta^{L+1} * df/dw; where f = w*x+b
    # Notice: X needs to be reshape
    X = X.reshape(X.shape[0], W.shape[0])
    dw = np.dot(X.T, dout)
    
    # 4. given the derivative of loss wrt to unit of one layer ahead, 
    # dl/db = delta^{L+1} * df/db; where f = w*x+b
    # !!!!!!!! notice, db needs to be sumed up 'class'-wise
    db = np.sum(dout, axis = 0)
    #db = dout
    
    return dx, dw, db
	
# Implement the forward pass for the ReLU activation function 
# in the relu_forward function
def relu_forward(X):
    ##############################################################
    #                                                            #
    # This is actually a unit for ReLu activation function.      #
    # out can be thinking as the output of this layer            #
    # cache can be thinking as the input of this layer           #
    #                                                            #
    ##############################################################
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    
    # 1. because this is a simple activation function in sake of modularization,
    # so we don't need to consider the matrix multiplication
    output = np.maximum(0, X)
    cache = X
    
    # 3. return outputs and intermediate information
    return output, cache
				
				
def relu_backward(dout, cache):
    ##################################################################
    #                                                                #
    # This is actually doing back propagation via back propagation.  #
    # dout is the derivative of loss wrt outputs of layer ahead      #
    # cache is the input of current layer, and it is stored by       #
    # affine_forward function.                                       #
    #                                                                #
    ##################################################################
    '''
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: - dout: derivative of loss wrt to unit of one layer ahead, of shape (N, C)
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    '''
    dx, X = None, cache
    
    # dL/dx = delta^{L+1} * dReLu/dx = delta^{L+1} where output from ReLu is > 0
    # while output from ReLu is < 0 dL/dx = delta^{L+1} * dReLu/dx = 0
    
    # two ways to implement: 1. create a binary indicator matrix and multiply to dout (inefficient)
    
    # binary_indicator = np.ones_like(X)
    # binary_indicator[X < 0] = 0
    # dx = dout * binary_indicator
    
    # 2. below
    dx = dout
    dx[cache < 0] = 0  
    
    return dx				
				
			


if __name__ == '__main__':
	num_inputs = 2
	input_shape = (4, 5, 6)
	output_dim = 3
	
	input_size = num_inputs * np.prod(input_shape)
	weight_size = output_dim * np.prod(input_shape)
	
	x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
	w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
	b = np.linspace(-0.3, 0.1, num=output_dim)
	
	# Test the affine_forward function

	out, _ = affine_forward(x, w, b)
	correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
	                        [ 3.25553199,  3.5141327,   3.77273342]])
	
	# Compare your output with ours. The error should be around 1e-9.
	print('Testing affine_forward function:')
	print('difference: ', rel_error(out, correct_out))
	
	# Test the affine_backward function
	x = np.random.randn(10, 2, 3)
	w = np.random.randn(6, 5)
	b = np.random.randn(5)
	dout = np.random.randn(10, 5)
	
	_, cache = affine_forward(x, w, b)
	dx, dw, db = affine_backward(dout, cache)
	
	dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
	dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
	db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
	
	# we first need to make sure analytic/numerical gradients are in the identical shapes
	print('analytic dx:', dx.shape, '/ numerical dx_num', dx_num.shape)
	print('analytic dw:', dw.shape, '/ numerical dw_num', dw_num.shape)
	print('analytic db:', db.shape, '/ numerical db_num', db_num.shape)
		
	dx_num = dx_num.reshape(dx_num.shape[0], -1)
	
	# The error should be around 1e-10
	print('Testing affine_backward function:')
	print('dx error: ', rel_error(dx_num, dx))
	print('dw error: ', rel_error(dw_num, dw))
	print('db error: ', rel_error(db_num, db))
	
	# Test the relu_forward function
	x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
	
	
	out, _ = relu_forward(x)
	correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
	                        [ 0.,          0.,          0.04545455,  0.13636364,],
	                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
	
	# Compare your output with ours. The error should be around 1e-8
	print('Testing relu_forward function:')
	print('difference: ', rel_error(out, correct_out))	
		
	# test ReLu_backward
	x = np.random.randn(10, 10)
	dout = np.random.randn(*x.shape)
	
	dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
	_, cache = relu_forward(x)
	dx = relu_backward(dout, cache)
	
	# The error should be around 1e-12
	print('Testing relu_backward function:')
	print('dx error: ', rel_error(dx_num, dx))
		
	
	
	
	
	
	
	
	
	
	
	
	