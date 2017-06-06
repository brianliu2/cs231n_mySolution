# -*- coding: utf-8 -*-
"""
There are some common patterns of layers that are frequently used in neural nets. 
For example, affine layers are frequently followed by a ReLU nonlinearity.

@author: xliu
"""

import numpy as np
from sep_forward_back_module import *
from batch_normalization import *
from dropout import *

def affine_relu_forward(X, W, b):
    '''
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    '''
    # f(W,X,b): (input, output) --> ReLu (input, output) --> Output --> backprop (output:gradients, input:(dout, inputs of ReLu))
    
    # 1. process forward --> store layer output and layer input ---> matrix multiplication
    outs_layer, inputs_layer = affine_forward(X, W, b)
    
    # 2. process through ReLu activation --> store layer output and ReLu input
    relu_output, relu_input = relu_forward(outs_layer)
    
    # 3. store inputs of linear multiplication layer and inputs of ReLu layer
    # actually outputs of linear multiplication == inputs of ReLu layer
    cache = (inputs_layer, relu_input)
    
    return relu_output, cache
				

def affine_relu_backward(dout, cache):
    '''
    Backward pass for the affine-relu convenience layer
    '''
    # Given delta_^{L+1} -> backprop ReLu (store outputs) -> backprop f() matrix multiplication (store dx, dw, db)
    # Notice that there is no inputs are stored in bakcward
    
    # 1. we retrieve inputs for matrix multiplication layer and ReLu layer
    matMul_input, relu_input = cache
    
    # 2. backprop the ReLu layer
    backprop_out_relu = relu_backward(dout, relu_input)
    
    # 3. backprop the matrix multiplication layer to get dx, dw, db
    dx, dw, db = affine_backward(backprop_out_relu, matMul_input)
    
    return dx, dw, db


def affine_bn_relu_forward(X, W, b, gamma, beta, bn_param):
	# 0. initialization	
	relu_out, matmul_bn_relu_input_info = 0, None
	
	# 1. matrix multiplication
	matmul_out, matmul_input_info = affine_forward(X, W, b)
	
	# 2. batch normalization
	bn_out, bn_input_info = batchnorm_forward_allQuantities(matmul_out,\
							  gamma, beta, bn_param)
	
	# 3. relu --> output
	relu_out, relu_input_info = relu_forward(bn_out)
	
	# 4. store inputs from three layers
	matmul_bn_relu_input_info = (matmul_input_info,\
								   bn_input_info, relu_input_info)
	
	return relu_out, matmul_bn_relu_input_info

def affine_bn_relu_backward(dout, three_layers_input_info):
	# 0. unpack all input information from three layers
	# matrix multiplication --> batch normalization --> ReLu
	matmul_input_info, bn_input_info, relu_input_info = three_layers_input_info
	
	# 1. backprob through relu
	drelu_layer = relu_backward(dout, relu_input_info)
	
	# 2. backprob through batch normalization layer
	dbn_layer_input, dgamma, dbeta = batchnorm_backward(drelu_layer, bn_input_info)
	
	# 3. backprob through matrix multiplication layer
	dX, dW, db = affine_backward(dbn_layer_input, matmul_input_info)
	
	# 4. return gradients
	return dX, dW, db, dgamma, dbeta

def affine_bn_relu_dropout_forward(X, W, b, gamma, beta, bn_param, dropout_param):
	# 1. initialize out and four layers input information tuple
	dropout_out, matmul_bn_relu_dropout_input_info = None, None
	
	# 2. layer_1: matrix multiplication
	matmul_out, matmul_input_info = affine_forward(X, W, b)
	
	# 3. layer_2: batch normalization
	bn_out, bn_input_info = batchnorm_forward_allQuantities(matmul_out, gamma, beta, bn_param)
	
	# 4. layer_3: ReLu
	relu_out, relu_input_info = relu_forward(bn_out)
	
	# 5. layer_4: dropout
	dropout_out, dropout_input_info = dropout_forward(relu_out, dropout_param)
	
	# 6. create input information tuple
	matmul_bn_relu_dropout_input_info = (matmul_input_info, bn_input_info,\
											relu_input_info, dropout_input_info)
	
	# 7. return outputs and tuple
	return dropout_out, matmul_bn_relu_dropout_input_info
	
def affine_bn_relu_dropout_backward(dout, four_layers_input_info):
	# 0. retrieve input info for all four layers
	matmul_input_info, bn_input_info, \
	relu_input_info, dropout_input_info = four_layers_input_info
	
	# 1. initialize grads
	dX, dW, db, dgamma, dbeta = None, None, None, None, None
	
	# 2. backprop through dropout
	dX_dropout = dropout_backward(dout, dropout_input_info)
	
	# 3. backprop through relu
	drelu = relu_backward(dX_dropout, relu_input_info)
	
	# 4. backprop through batch normalization
	dbatch_norm, dgamma, dbeta = batchnorm_backward(drelu, bn_input_info)
	
	# 5. backprop through matri multiplication
	dX, dW, db = affine_backward(dbatch_norm, matmul_input_info)
	
	return dX, dW, db, dgamma, dbeta

if __name__ == '__main__':
	x = np.random.randn(2, 3, 4)
	w = np.random.randn(12, 10)
	b = np.random.randn(10)
	dout = np.random.randn(2, 10)
	
	# test the affine_relu_forward and affine_relu_backward
	out, cache = affine_relu_forward(x, w, b)
	dx, dw, db = affine_relu_backward(dout, cache)
	
	dx, dw, db = affine_relu_backward(dout, cache)

	dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
	dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
	db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)
	
	print('analytic dx:', dx.shape, '/ numerical dx_num', dx_num.shape)
	print('analytic dw:', dw.shape, '/ numerical dw_num', dw_num.shape)
	print('analytic db:', db.shape, '/ numerical db_num', db_num.shape)
	
	dx_num = dx_num.reshape(dx_num.shape[0], -1)
	
	print('Testing affine_relu_forward:')
	print('dx error: ', rel_error(dx_num, dx))
	print('dw error: ', rel_error(dw_num, dw))
	print('db error: ', rel_error(db_num, db))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	