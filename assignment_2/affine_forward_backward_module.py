# -*- coding: utf-8 -*-
"""
There are some common patterns of layers that are frequently used in neural nets. 
For example, affine layers are frequently followed by a ReLU nonlinearity.

@author: xliu
"""

import numpy as np
from sep_forward_back_module import *

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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	