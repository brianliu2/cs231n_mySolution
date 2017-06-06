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
from batch_normalization import *
#from cs231n.layer_utils import *
from dropout import *

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
		
		# initialize the hyper-parameters of batch_normalization
		# because the batch normalization will be only used along activation unit
		# This indicates that only hidden layers with activation functions need
		# to initialize hyper-parameters for batch_norm. 
		# The first matrix multiplication and 
		# last score evaluation layer (matrix multiplication, actually) have no 
		# batch normalization layer, of course they don't need to initialize 
		# hyper parameters.
		
		
		# for each layer with activation, it has own batch normalization hyper-parameters
		# bn_param: {'mode' --> (train, test); 
		#            'running_mean' --> R: (D, )
		#            'running_variance' --> R:(D, )
		#           }
		# The reason for storing mean and variance evaluated for every iteration
		# at each layer performed batch normalization, is due to when we perform
		# prediction of unknown incoming data, we want to normalize those unknown
		# incoming data in the identical way we perform during training stage
		# Momentum scheme is used in terms of cooling the pertubation along iteration
		# rolls
		#
		# gamma: (D, )
		# beta: (D, )
		if self.use_batch_normal:
			print('batch_normal is inserted before activation layer.')
			
			
			# for each layer with activation function, it has own batch normalization
			# settings. We save all bn_param in self.container as dictionary contained dictionaries
			# 
			# bn_params = {'bn_param1', 'bn_param2', 'bn_param3', ..., 'bn_paramN'}
			# 'bn_params1': {'mode': 'train', 'running_mean': zeros(D_1), 'running_variance': zeros(D_1)}
			# 'bn_params2': {'mode': 'train', 'running_mean': zeros(D_2), 'running_variance': zeros(D_2)}
			# 'bn_params3': {'mode': 'train', 'running_mean': zeros(D_3), 'running_variance': zeros(D_3)}
			# .
			# .
			# .
			# 'bn_paramsN': {'mode': 'train', 'running_mean': zeros(D_N), 'running_variance': zeros(D_N)}
			self.bn_param_w_activation_layers = {'bn_param'+str(l):
													   {'mode': 'train',
													    'running_mean': np.zeros(dims[l]),
													    'running_variance': np.zeros(dims[l])}
				 for l in range(1, len(dims)-1)}
			
			# because this is variance of standard Gaussian, so gamma is initialize
			# as one
			gamma_s = {'gamma'+str(l): np.ones(dims[l]) for l in range(1, len(dims)-1)}
			
			# because this is mean of standard Gaussian, so beta is initialize
			# as zero
			beta_s = {'beta'+str(l): np.zeros(dims[l]) for l in range(1, len(dims)-1)}
			
			self.params.update(gamma_s)
			self.params.update(beta_s)
		
		## ------------- initialization to dropout layer -------------- ##
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {'mode': 'train', 'perc': dropout}
			if seed is not None:
				self.dropout_param['seed'] = seed
		
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
				if self.use_dropout:
					init_x_dropout_out, init_x_dropout_input_info = dropout_forward(\
																layer_in_out_grads['L'+str(n-1)+'_out'],\
																self.dropout_param)
					layer_in_out_grads['L'+str(n)+'_dropout_out'] = init_x_dropout_out
					layer_in_out_grads['L'+str(n)+'_dropout_input_info'] = init_x_dropout_input_info
					L_pre_out = init_x_dropout_out
					layer_in_out_grads['L'+str(n)+'_input_info'] = (L_pre_out, W, b)
				else:
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
				if self.use_batch_normal:
					gamma = self.params['gamma' + str(n)]
					beta  = self.params['beta'  + str(n)]
					bn_param = self.bn_param_w_activation_layers['bn_param'+str(n)]
					
					L_out, L_in = affine_bn_relu_forward(L_pre_out, W, b, gamma, beta, bn_param)
					layer_in_out_grads['L'+str(n)+'_out'] = L_out
					layer_in_out_grads['L'+str(n)+'_input_info'] = L_in
				else:
					L_out, L_in = affine_relu_forward(L_pre_out, W, b)
					layer_in_out_grads['L'+str(n)+'_out'] = L_out
					layer_in_out_grads['L'+str(n)+'_input_info'] = L_in
				# In forward pass, we do batch normalization --> dropout
				# In backward pass, we do dropout_back --> batchnorm_back
				if self.use_dropout:
					L_dropout_out, L_dropout_in = dropout_forward(L_out, self.dropout_param)
					layer_in_out_grads['L'+str(n)+'_dropout_out'] = L_dropout_out
					layer_in_out_grads['L'+str(n)+'_dropout_input_info'] = L_dropout_in
					
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
				# dropout
				if self.use_dropout:
					layer_dropout_input_info = layer_in_out_grads['L'+str(n)+'_dropout_input_info']
					grads_layer_ahead = layer_in_out_grads['d_L'+str(n+1)+'_X']
					
					d_dropout_x = dropout_backward(grads_layer_ahead, layer_dropout_input_info)
				# batch_norm
				if self.use_batch_normal:
					layer_input_info = layer_in_out_grads['L'+str(n)+'_input_info']
					if self.use_dropout:
						grads_layer_ahead = d_dropout_x
					else:
						grads_layer_ahead = layer_in_out_grads['d_L'+str(n+1)+'_X']
					
					dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(grads_layer_ahead, layer_input_info)
					layer_in_out_grads['d_L'+str(n)+'_X'] = dx
					layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] = dw
					layer_in_out_grads['d_L'+str(n)+'_b'+str(n)] = db
					layer_in_out_grads['d_L'+str(n)+'_gamma'+str(n)] = dgamma
					layer_in_out_grads['d_L'+str(n)+'_beta'+str(n)] = dbeta
					# dont't forget the regularization penalty
					layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] += self.reg * self.params['W'+str(n)]
					
				else:
					# no batch normal and no dropout
					layer_input_info = layer_in_out_grads['L'+str(n)+'_input_info']
					
					if self.use_dropout:
						grads_layer_ahead = d_dropout_x
					else:
						grads_layer_ahead = layer_in_out_grads['d_L'+str(n+1)+'_X']
						
					grads_layer_ahead = layer_in_out_grads['d_L'+str(n+1)+'_X']
					dx,dw,db = affine_relu_backward(grads_layer_ahead, layer_input_info)
					layer_in_out_grads['d_L'+str(n)+'_X'] = dx
					layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] = dw
					layer_in_out_grads['d_L'+str(n)+'_b'+str(n)] = db
					# dont't forget the regularization penalty
					layer_in_out_grads['d_L'+str(n)+'_W'+str(n)] += self.reg * self.params['W'+str(n)]
				
		
		# 5.4 create two dictionarys to store dW and db
		grads = {}
		grads_w = {key[-2:]: val for key, val in layer_in_out_grads.items() if key[-2]=='W'}
		grads_b = {key[-2:]: val for key, val in layer_in_out_grads.items() if key[-2]=='b'}
		
		if self.use_batch_normal:
			grads_gamma = {key[-6:]: val for key, val in layer_in_out_grads.items() if key[-6:-1]=='gamma'}
			grads_beta  = {key[-5:]: val for key, val in layer_in_out_grads.items() if key[-5:-1]=='beta'}
			grads.update(grads_gamma)
			grads.update(grads_beta)
			
		# 5.8 update grads dictionary
		grads.update(grads_w)
		grads.update(grads_b)
																
		# 6. return all results
																
		return loss, grads



#	def run_model(hidden_dims, input_dims, num_classes, **kwargs):
#		if optim_config is None:
#			raise ValueError('optim_config can not be empty')	
#		optim_config={'learning_rate': learning_rate,}
#		weight_scale = kwargs.get('weight_scale', 1e-2)
#		num_epochs   = kwargs.get('num_epochs', 20)
#		batch_size   = kwargs.get('batch_size', 25)
#		update_rule  = kwargs.get('update_rule', 'sgd')
#		print_every  = kwargs.get('print_every', 10)
#		model = fullyConnectedNet(hidden_dims, input_dims, num_classes, weight_scale = weight_scale)
#	#	Solver = solver(model, data, print_every = print_every, 
#	#					 num_epochs = num_epochs, batch_size = batch_size, update_rule = update_rule,
#	#					 optim_config = optim_config)
#		Solver = solver(model, data, print_every, 
#						 num_epochs, batch_size, update_rule,
#						 optim_config)
#		Solver.train()
#		return Solver.train_err_history

def run_model(data, hidden_dims,input_dims, num_classes, weight_scale,learning_rate):
    model = fullyConnectedNet(hidden_dims, input_dims, num_classes,
            weight_scale=weight_scale, dtype=np.float64)
    Solver = solver(model, data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                    'learning_rate': learning_rate,
                    }
             )
    Solver.train()
    return Solver.loss_history, Solver.train_err_history

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
	
#	N, D, H1, H2, H3, H4, C = 2, 15, 20, 30, 40, 50, 10
#	X = np.random.randn(N, D)
#	y = np.random.randint(C, size=(N,))
#	
#	for reg in [0, 3.14]:
#	  print('Running check with reg = ', reg)
#	  model = fullyConnectedNet([H1, H2, H3, H4], input_dims=D, num_classes=C,\
#			   reg=reg, weight_scale=5e-2, dtype=np.float64, use_batch_normal = False)
#
#	  loss, grads = model.loss(X, y)
#	  print('Initial loss: ', loss)
#
#	  for name in sorted(model.params.keys()):
#				f = lambda _: model.loss(X, y)[0]
#				grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#				print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#
#
#
#	num_train = 50
#	small_data = {
#	  'X_train': data['X_train'][:num_train],
#	  'y_train': data['y_train'][:num_train],
#	  'X_val': data['X_val'],
#	  'y_val': data['y_val'],
#	}
#	
#	weight_scale = 1e-3
#	learning_rate = 1e-5
#	model = fullyConnectedNet([100, 100, 100, 100], input_dims = 3 * 32 * 32, num_classes = 10,
#	              weight_scale=weight_scale, dtype=np.float64)
#	Solver = solver(model, small_data,
#	                print_every=10, num_epochs=20, batch_size=25,
#	                update_rule='sgd',
#	                optim_config={
#	                  'learning_rate': learning_rate,
#	                }
#	         )
#	Solver.train()
#	
#	plt.plot(Solver.loss_history, 'o')
#	plt.title('Training loss history')
#	plt.xlabel('Iteration')
#	plt.ylabel('Training loss')
#	plt.show()
	
	


## --- test for batch_norm activated fully connect NN
#	np.random.seed(231)
#	N, D, H1, H2, C = 2, 15, 20, 30, 10
#	X = np.random.randn(N, D)
#	y = np.random.randint(C, size=(N,))
#	
#	for reg in [0, 3.14]:
#	  print('Running check with reg = ', reg)
#	  model = fullyConnectedNet([H1, H2], input_dims=D, num_classes=C,
#	                            reg=reg, weight_scale=5e-2, dtype=np.float64,
#	                            use_batch_normal=True)
#	
#	  loss, grads = model.loss(X, y)
#	  print('Initial loss: ', loss)
#	
#	  for name in sorted(grads):
#	    f = lambda _: model.loss(X, y)[0]
#	    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#	    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#	  if reg == 0: print()



## --- test for more hidden layers batch norm activated NN
	np.random.seed(231)
	# Try training a very deep net with batchnorm
	hidden_dims = [100, 100, 100, 100, 100]
	
	num_train = 1000
	small_data = {
	  'X_train': data['X_train'][:num_train],
	  'y_train': data['y_train'][:num_train],
	  'X_val': data['X_val'],
	  'y_val': data['y_val'],
	}
	
	weight_scale = 2e-2
	bn_model = fullyConnectedNet(input_dims = 32 * 32 * 3, hidden_dims = hidden_dims, num_classes = 10, \
				weight_scale=weight_scale, use_batch_normal=True)
	
	model = fullyConnectedNet(input_dims = 32 * 32 * 3, hidden_dims = hidden_dims, num_classes = 10, \
				weight_scale=weight_scale, use_batch_normal=False)
	
	bn_solver = solver(bn_model, small_data,
	                num_epochs=10, batch_size=50,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 1e-3,
	                },
	                verbose=True, print_every=200)
	bn_solver.train()
	
	solver = solver(model, small_data,
	                num_epochs=10, batch_size=50,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 1e-3,
	                },
	                verbose=True, print_every=200)
	solver.train()


	plt.subplot(3, 1, 1)
	plt.title('Training loss')
	plt.xlabel('Iteration')
	
	plt.subplot(3, 1, 2)
	plt.title('Training error')
	plt.xlabel('Epoch')
	
	plt.subplot(3, 1, 3)
	plt.title('Validation error')
	plt.xlabel('Epoch')
	
	plt.subplot(3, 1, 1)
	plt.plot(solver.loss_history, 'o', label='baseline')
	plt.plot(bn_solver.loss_history, 'o', label='batchnorm')
	
	plt.subplot(3, 1, 2)
	plt.plot(solver.train_err_history, '-o', label='baseline')
	plt.plot(bn_solver.train_err_history, '-o', label='batchnorm')
	
	plt.subplot(3, 1, 3)
	plt.plot(solver.valid_err_history, '-o', label='baseline')
	plt.plot(bn_solver.valid_err_history, '-o', label='batchnorm')
	  
	for i in [1, 2, 3]:
	  plt.subplot(3, 1, i)
	  plt.legend(loc='upper center', ncol=4)
	plt.gcf().set_size_inches(15, 15)
	plt.show()



#	'''
#	Batch normalization and initialization
#	We will now run a small experiment to study the interaction of batch 
#	normalization and weight initialization.
#	The first cell will train 8-layer networks both with and without 
#	batch normalization using different scales for weight initialization. 
#	The second layer will plot training accuracy, validation set accuracy, 
#	and training loss as a function of the weight initialization scale.
#	'''
#	np.random.seed(231)
#	# Try training a very deep net with batchnorm
#	hidden_dims = [50, 50, 50, 50, 50, 50, 50]
#	num_train = 1000
#	small_data = {
#	  'X_train': data['X_train'][:num_train],
#	  'y_train': data['y_train'][:num_train],
#	  'X_val': data['X_val'],
#	  'y_val': data['y_val'],
#	}
#	bn_solvers = {}
#	Solvers = {}
#	weight_scales = np.logspace(-4, 0, num=20)
#	for i, weight_scale in enumerate(weight_scales):
#		print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
#		bn_model = fullyConnectedNet(input_dims = 32 * 32 * 3, hidden_dims = hidden_dims, num_classes = 10, \
#		weight_scale=weight_scale, use_batch_normal=True)
#		
#		model = fullyConnectedNet(input_dims = 32 * 32 * 3, hidden_dims = hidden_dims, num_classes = 10, \
#		weight_scale=weight_scale, use_batch_normal=False)
#		
#		bn_solver = solver(bn_model, small_data, num_epochs=10, batch_size=50,update_rule='adam',\
#		optim_config={'learning_rate': 1e-3}, verbose=False, print_every=200)
#		
#		bn_solver.train()
#		bn_solvers[weight_scale] = bn_solver
#		
#		Solver = solver(model, small_data,num_epochs=10, batch_size=50,update_rule='adam',\
#		optim_config={'learning_rate': 1e-3},verbose=False, print_every=200)
#		
#		Solver.train()
#		Solvers[weight_scale] = Solver
#	
#	# Plot results of weight scale experiment
#	best_train_accs, bn_best_train_accs = [], []
#	best_valid_accs, bn_best_valid_accs = [], []
#	final_train_loss, bn_final_train_loss = [], []
#	
#	for ws in weight_scales:
#	  best_train_accs.append(min(Solvers[ws].train_err_history))
#	  bn_best_train_accs.append(min(bn_solvers[ws].train_err_history))
#	  
#	  best_valid_accs.append(min(Solvers[ws].valid_err_history))
#	  bn_best_valid_accs.append(min(bn_solvers[ws].valid_err_history))
#	  
#	  final_train_loss.append(np.mean(Solvers[ws].loss_history[-100:]))
#	  bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))
#	  
#	plt.subplot(3, 1, 1)
#	plt.title('Best val accuracy vs weight initialization scale')
#	plt.xlabel('Weight initialization scale')
#	plt.ylabel('Best val accuracy')
#	plt.semilogx(weight_scales, best_valid_accs, '-o', label='baseline')
#	plt.semilogx(weight_scales, bn_best_valid_accs, '-o', label='batchnorm')
#	plt.legend(ncol=2, loc='lower right')
#	
#	plt.subplot(3, 1, 2)
#	plt.title('Best train accuracy vs weight initialization scale')
#	plt.xlabel('Weight initialization scale')
#	plt.ylabel('Best training accuracy')
#	plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
#	plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
#	plt.legend()
#	
#	plt.subplot(3, 1, 3)
#	plt.title('Final training loss vs weight initialization scale')
#	plt.xlabel('Weight initialization scale')
#	plt.ylabel('Final training loss')
#	plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
#	plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
#	plt.legend()
#	plt.gca().set_ylim(1.0, 3.5)
#	
#	plt.gcf().set_size_inches(10, 15)
#	plt.show()
	











































