# -*- coding: utf-8 -*-
"""
It is different from the implementation of a two-layer neural network in a 
single monolithic class. With having modular versions of the necessary layers, 
we can reimplement the two layer network using these modular implementations.

class object of modularized two layers neural network

@author: xliu
"""
import numpy as np
from affine_forward_backward_module import *
from sep_forward_back_module import *
from loss_svm_softmax import *

class twoLayerModularNN(object):
    '''
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
  
    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    '''
    
    # the initial settings: dim_input, num_hidden, num_class, weight_scale, reg 
    def __init__(self, dim_input, num_hidden, num_class, weight_scale, reg):
        ############################################################################
        # Initialize the weights and biases of the two-layer net. Weights          #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(dim_input, num_hidden)
        self.params['b1'] = np.zeros(num_hidden)
        self.params['W2'] = weight_scale * np.random.randn(num_hidden, num_class)
        self.params['b2'] = np.zeros(num_class)
        self.reg = reg
    
    # define the function to evaluate the loss value
    def loss(self, X, y):
        '''        
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        '''
        ## 0. retrieve weights and bias information from self object
        #W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        
        # 1. evaluate scores based on given X and weight
        scores = 0      
        
        # -- 1.1 modularized matrix_multiplication_1_plus_bias_1 -> ReLu -> return: relu_output, (matmul_input, relu_input)
        hidden_relu_out, hidden_relu_matmul_inputs = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        
        # -- 1.2 modularized matrix_multiplication_2_plus_bias_2 -> return: matmul_output, (matmul_input, matmul_W, matmul_b)
        output_layer_out, output_layer_inputs = affine_forward(hidden_relu_out, self.params['W2'], self.params['b2'])
        
        # -- 1.3 if there is no output information, we only return the scores (probabilities from softmax)
        if y is None:
            scores = output_layer_out
            return scores
        
        # -------------------------------------------------------------------------------------------------#
        
        # 2. evluate the loss and gradient if label y is provided
        
        # -- 2.1 if there is output information, we return tuple = (loss, grads)
        data_loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the two-layer net. Store the loss        #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # -- 2.2 evaluate loss and gradient at the dloss/module_2_output
        data_loss, dout = softmax_loss(output_layer_out, y)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # don't forget the regularization cost
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) \
                                    + np.sum(self.params['W2'] * self.params['W2']))
        data_loss += reg_loss
        
        
        # -- 2.3 flow back to last hidden layer: 
        #    dloss/dhidden_input = dloss/dmodule_2_output * dmodule_2_output/dhidden_input
        #    dloss/dhidden_weight = dloss/dmodule_2_output * dmodule_2_output/dhidden_weight
        #    dloss/dhidden_bias = dloss/dmodule_2_output * dmodule_2_output/dhidden_bias
        
        #    dloss/dhidden_input is used as delta^{L+1} to be flowed back to backprop of module_(L)
        dhidden_intput, dhidden_weight, dhidden_bias = affine_backward(dout, output_layer_inputs)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # -- 2.4 don't forget the regularization cost
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dhidden_weight += self.reg * self.params['W2']
        
        # -- 2.5 flow back to last hidden layer: 
        #    dloss/dinput = dloss/dmodule_2_output * dmodule_2_output/dmodule_2_input * dmodule_2_input/dinput
        #    dloss/dweight = dloss/dmodule_2_output * dmodule_2_output/dmodule_2_input * dmodule_2_input/dweight
        #    dloss/dbias = dloss/dmodule_2_output * dmodule_2_output/dmodule_2_input * dmodule_2_input/dbias
        dinput, dweight, dbias = affine_relu_backward(dhidden_intput, hidden_relu_matmul_inputs)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # -- 2.6 don't forget the regularization cost
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dweight += self.reg * self.params['W1']
        
        # update all grads
        grads.update({'W1': dweight,
                      'b1': dbias,
                      'W2': dhidden_weight,
                      'b2': dhidden_bias})
        
        return data_loss, grads
								
								
								
								
if __name__ == '__main__':
	
	# sanity check of implementation
	
	N, D, H, C = 3, 5, 50, 7
	X = np.random.randn(N, D)
	y = np.random.randint(C, size=N)
	
	std = 1e-2
	model = twoLayerModularNN(dim_input=D, num_hidden=H, num_class=C, weight_scale=std, reg=0)							

	print('Testing initialization ... ')
	W1_std = abs(model.params['W1'].std() - std)
	b1 = model.params['b1']
	W2_std = abs(model.params['W2'].std() - std)
	b2 = model.params['b2']

	assert W1_std < std / 10, 'First layer weights do not seem right'
	assert np.all(b1 == 0), 'First layer biases do not seem right'
	assert W2_std < std / 10, 'Second layer weights do not seem right'
	assert np.all(b2 == 0), 'Second layer biases do not seem right'
	
	print('Testing test-time forward pass ... ')
	model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
	model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
	model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
	model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
	
	X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
	scores = model.loss(X, None)
	
	correct_scores = np.asarray(
	[[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
	[12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
	[12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
	
	scores_diff = np.abs(scores - correct_scores).sum()
	assert scores_diff < 1e-6, 'Problem with test-time forward pass'
	
	print('Testing training loss (no regularization)')
	y = np.asarray([0, 5, 1])
	loss, grads = model.loss(X, y)
	correct_loss = 3.4702243556
	print(abs(loss - correct_loss))
	assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
	
	model.reg = 1.0
	loss, grads = model.loss(X, y)
	correct_loss = 26.5948426952
	print(abs(loss - correct_loss))
	assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
		
	for reg in [0.0, 0.7]:
	    print('Running numeric gradient check with reg = ', reg)
	    model.reg = reg
	    loss, grads = model.loss(X, y)
	    
	    for name in sorted(grads):
	        f = lambda _: model.loss(X, y)[0]
	        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
	        print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
		

	

	
	










