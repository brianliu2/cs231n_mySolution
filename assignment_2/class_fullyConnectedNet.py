# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:11:24 2017

@author: xliu
"""


import numpy as np

from affine_forward_backward_module import *
from sep_forward_back_module import *
from loss_svm_softmax import *
from solver import *
import matplotlib.pyplot as plt
#from cs231n.layer_utils import *

class fullyConnectedNet(object):
	'''
	A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
	'''
	def __init__(self, hidden_dims, input_dims, num_classes, weight_scale,
				  dropout=0, use_batch_normal = False, reg = 0.0, dtype=np.float32,
				  seed = None):
		'''
		Initialize a new network.
		Inputs:
		- input_dim: An integer giving the size of the input
		- hidden_dim: An integer giving the size of the hidden layer
		- num_classes: An integer giving the number of classes to classify
		- dropout: Scalar between 0 and 1 giving dropout strength.
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		'''
		self.use_batch_normal = use_batch_normal
		self.use_dropout  = dropout > 0
		self.reg = reg
		self.dtype = dtype
		
		# number of layers
		self.num_layers = len(hidden_dims) + 1
		
		# create a dictionary to store parameters
		self.params = {}
		
		# assign dimension information
		self.D = input_dims
		self.C = num_classes
		
		#############################################################################
        # Initialize the parameters of the network, storing all values in          #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
		
		# 2. make sure hidden_dims is a list, otherwise throw an error
		if type(hidden_dims) != list:
			raise ValueError('hidden_dims has to be a list')
		
		# 3. initialize weights based on their different dimensions
		params = {}
		
		# 3.1 -- because the dimension information looks like [10, [20, 30, 40], 50]
		#        we need to manually convert to [10, 20, 30, 40, 50]
		dims = [[self.D], hidden_dims, [self.C]]
		
		# 3.1.1 -- flatten the list
		dims = [dim for subdims in dims for dim in subdims]
		
		# 3.2 -- initialize neural network parameters
#		Ws = {}
#		bs = {}
#		for n in range(self.num_layers-1):
#			Ws['W' + str(n+1)]=weight_scale * np.random.randn(dims[n],dims[n+1])
#			bs['b' + str(n+1)]=np.zeros(dims[n+1])
		
		# 3.2.1 -- more 'python' way to write loop
		Ws = {'W' + str(n+1): weight_scale * np.random.randn(dims[n],dims[n+1])\
			  for n in range(len(dims)-1)}
		bs = {'b' + str(n+1): np.zeros(dims[n+1]) for n in range(len(dims)-1)}	
		
		# 3.3 -- pop in the parameter dictionary
		self.params.update(bs)
		self.params.update(Ws)
		
		 #--------------------------------------------------------------------------#
        #                                                                          #
        #                                                                          #
        # Missing part 1: batch normalization                                      #
        #                                                                          #
        #                                                                          #
        #                                                                          #
        #                                                                          #
        # Missing part 2: dropout                                                  #
        #                                                                          #
        #                                                                          #
        #--------------------------------------------------------------------------#
		
	##############################################################
	#
	#
	#	loss function
	#	
	#
	##############################################################
	def loss(self, X, y):
		'''
		Compute loss and gradient for the fully-connected net.

		Input / output: Same as TwoLayerNet.
		'''
		# 1. convert input X into the dtype we want
		X = X.astype(self.dtype)
		
		# 2. determine the mode of running this loss function, whether we 
		# want to test its correctness or really perform to evaluate loss/grads
		if y is None:
			mode = 'test'
		else:
			mode = 'train'
		
		# 3. initialize scores
		scores = 0
		
		############################################################################
		# Implement the forward pass for the fully-connected net, computing        #
		# the class scores for X and storing them in the scores variable.          #
		#                                                                          #
		# When using dropout, you'll need to pass self.dropout_param to each       #
		# dropout forward pass.                                                    #
		#                                                                          #
		# When using batch normalization, you'll need to pass self.bn_params[0] to #
		# the forward pass for the first batch normalization layer, pass           #
		# self.bn_params[1] to the forward pass for the second batch normalization #
		# layer, etc.                                                              #
		############################################################################
		
		# 4. proceed forward to evaluate scores (pre-output, output needs to be np.argmax() )		
		# store everything in a dictionnary hidden
		layer_in_out_grads = {}
		
		# input of hidden layer 0 is original input, but here we need to reshape inputs
		X = np.reshape(X, (X.shape[0], -1))
		layer_in_out_grads['L0_out'] = X
		# 4.1 loop-over all layers
		for n in range(1, self.num_layers+1):
			
			W = self.params['W'+str(n)]
			b = self.params['b'+str(n)]
			
			# if current layer is first forward layer, we set layer input to
			# original X
			if n == 1:
				L_pre_out = layer_in_out_grads['L'+str(n-1)+'_out']
				#layer_in_out_grads['L'+str(n)+'_out'] = X
				layer_in_out_grads['L'+str(n)+'_input_info'] = (L_pre_out, W, b)
			else:
				# (n-1)_out = (n)_in
				L_pre_out = layer_in_out_grads['L'+str(n-1)+'_out']
				
			# if the current layer is the last pre-output layer, then we
			# simply do matrix multiplication
			if n == self.num_layers:
				L_out, L_in = affine_forward(L_pre_out, W, b)
				layer_in_out_grads['L'+str(n)+'_out'] = L_out
				layer_in_out_grads['L'+str(n)+'_input_info'] = L_in
			# otherwise, we perform a modularized forward function
			# modular forward = (affine_forward + relu_forward)
			else:
				L_out, L_in = affine_relu_forward(L_pre_out, W, b)
				layer_in_out_grads['L'+str(n)+'_out'] = L_out
				layer_in_out_grads['L'+str(n)+'_input_info'] = L_in
		
		# 4.2 store scores value as the output from last layer
		scores = layer_in_out_grads['L'+str(self.num_layers)+'_out']
		
		# 4. 1if the mode is test, we then return scores
		if mode == 'test':
			return scores
		
		# if we really want to perform model training
		# -- we then need to perform loss function to evaluate loss value and 
		#    grads at the last layer, details can be found at class_twoLayerModularNN.py
		
		############################################################################
		# Implement the backward pass for the fully-connected net. Store the       #
		# loss in the loss variable and gradients in the grads dictionary. Compute #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		#                                                                          #
		# When using batch normalization, you don't need to regularize the scale   #
		# and shift parameters.                                                    #
		#                                                                          #
		# NOTE: To ensure that your implementation matches ours and you pass the   #
		# automated tests, make sure that your L2 regularization includes a factor #
		# of 0.5 to simplify the expression for the gradient.                      #
		############################################################################		
		
		# 5.1 compute the data_loss and the gradient of loss wrt the previous layer output
		data_loss, dscores = softmax_loss(scores, y)
		
		reg_loss = 0
		for w in [self.params[key] for key in self.params.keys() if key[0] == 'W']:
			reg_loss += 0.5 * self.reg * np.sum(w * w)
		
		loss = data_loss + reg_loss
		# 5.2 add all regularization penalties to data_loss
#		for w in [self.params[key] for key in self.params.keys() if key[0] == 'W']:
#			data_loss += 0.5 * self.reg * np.sum(w * w)
		
		#--------------------------------------
		#	
		#	
		#	5.3 back-prop:
		# 	if we have 5 layers (4 hidden + 1 output)
		#   we should have 6 grads: dh5 = dloss/dh5
		#	this layer is a loss evaluate layer, it is excluded from
		#	hidden + output layers.
		# 	Other than the loss grad, there are 5 grads: dh4, dh3, dh2, dh1, dh0
		# 	where dh4 is the gradient of output layer (use affine_back)
		#   dh3-dh0 (4 grads) use modularized affine_relu_back
		#
		#
		#--------------------------------------
		
		# 5.3.1 initialize the gradient of loss layer
		# --- notice that: this layer doen't count in num_layers (hidden layers + output layer)
		#layer_in_out_grads['d_L'+str(self.num_layers)] = dscores
		
		
		# 5.3.2 loop-over layers reversely
		for n in reversed(range(1, self.num_layers+1)):
			# 5.3.3 if the current layer is last output layer
			#       we then use affine_back function
			if n == (self.num_layers):
				layer_input_info = layer_in_out_grads['L'+str(n)+'_input_info']
				grads_layer_ahead = dscores
				dx,dw,db = affine_backward(grads_layer_ahead, layer_input_info)
				layer_in_out_grads['d_L'+str(n)+'_X'] = dx
				layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] = dw
				layer_in_out_grads['d_L'+str(n)+'_b'+str(n)] = db
				
				# dont't forget the regularization penalty
				layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] += self.reg * self.params['W'+str(n)]
			else:
				layer_input_info = layer_in_out_grads['L'+str(n)+'_input_info']
				grads_layer_ahead = layer_in_out_grads['d_L'+str(n+1)+'_X']
				dx,dw,db = affine_relu_backward(grads_layer_ahead, layer_input_info)
				layer_in_out_grads['d_L'+str(n)+'_X'] = dx
				layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] = dw
				layer_in_out_grads['d_L'+str(n)+'_b'+str(n)] = db
				# dont't forget the regularization penalty
				layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] += self.reg * self.params['W'+str(n)]
				
		
		# 5.4 create two dictionarys to store dW and db
		grads_w = {key[-2:]: val for key, val in layer_in_out_grads.items() if key[-2]=='W'}
		grads_b = {key[-2:]: val for key, val in layer_in_out_grads.items() if key[-2]=='b'}
		
		# 5.8 update grads dictionary
		grads = {}
		grads.update(grads_w)
		grads.update(grads_b)
																
		# 6. return all results
																
		return loss, grads




##############################################################
#
#
#	self-check
#
#
##############################################################
if __name__ == '__main__':
#	N, D, H1, H2, C = 3, 5, 30, 50, 7
#	X = np.random.randn(N, D)
#	y = np.random.randint(C, size=N)
#	
#	std = 1e-2
#	model = fullyConnectedNet(hidden_dims=[H1,H2], input_dims=D, num_classes=C, weight_scale=std, reg=0.5)							
#
#	print('Testing initialization ... ')
#	W1_std = abs(model.params['W1'].std() - std)
#	b1 = model.params['b1']
#	W2_std = abs(model.params['W2'].std() - std)
#	b2 = model.params['b2']
#	W3_std = abs(model.params['W3'].std() - std)
#	b3 = model.params['b3']
#
#
#	assert W1_std < std / 10, 'First layer weights do not seem right'
#	assert np.all(b1 == 0), 'First layer biases do not seem right'
#	assert W2_std < std / 10, 'Second layer weights do not seem right'
#	assert np.all(b2 == 0), 'Second layer biases do not seem right'
#	assert W3_std < std / 10, 'Second layer weights do not seem right'
#	assert np.all(b3 == 0), 'Second layer biases do not seem right'
#	
#	print('Testing test-time forward pass ... ')
#	X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
#	scores = model.loss(X, None)
#	
#	print('scores are correctly calculated')
#	
#	print('Testing training loss (no regularization)')
#	
#	y = np.asarray([0, 5, 1])
#	loss, grads = model.loss(X, y)
#	correct_loss = 3.4702243556
#	print(abs(loss - correct_loss))
#	#assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
	
	N, D, H1, H2, H3, H4, C = 2, 15, 20, 30, 40, 50, 10
	X = np.random.randn(N, D)
	y = np.random.randint(C, size=(N,))
	
	for reg in [0, 3.14]:
	  print('Running check with reg = ', reg)
	  model = fullyConnectedNet([H1, H2, H3, H4], input_dims=D, num_classes=C,reg=reg, weight_scale=5e-2, dtype=np.float64)

	  loss, grads = model.loss(X, y)
	  print('Initial loss: ', loss)

	  for name in sorted(model.params.keys()):
				f = lambda _: model.loss(X, y)[0]
				grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
				print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))



	num_train = 50
	small_data = {
	  'X_train': data['X_train'][:num_train],
	  'y_train': data['y_train'][:num_train],
	  'X_val': data['X_val'],
	  'y_val': data['y_val'],
	}
	
	weight_scale = 1e-3
	learning_rate = 1e-5
	model = fullyConnectedNet([100, 100, 100, 100], input_dims = 3 * 32 * 32, num_classes = 10,
	              weight_scale=weight_scale, dtype=np.float64)
	Solver = solver(model, small_data,
	                print_every=10, num_epochs=20, batch_size=25,
	                update_rule='sgd',
	                optim_config={
	                  'learning_rate': learning_rate,
	                }
	         )
	Solver.train()
	
	plt.plot(Solver.loss_history, 'o')
	plt.title('Training loss history')
	plt.xlabel('Iteration')
	plt.ylabel('Training loss')
	plt.show()
	
	

























































