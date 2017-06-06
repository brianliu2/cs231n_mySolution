# -*- coding: utf-8 -*-
"""
Batch normal is an additional layer inserted between matrix multiplication and
activation layer, so in order to flow forward/backprop, we need to write up it
base forward and backprop such that it can be incorrporated into a modularized
version.

@author: xliu
"""
import numpy as np
from gradient_check import * 

def batchnorm_forward(X, gamma, beta, bn_param):
	'''
	Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
	'''
	# 1. read-in algorithmic settings from arguments
	eps = bn_param.get('eps', 1e-5) # -- numerical stabilizer
	momentum = bn_param.get('momentum', 0.9)
	
	# 1.1 we have two modes for performing batch normalization:
	# --  training mode:
	# --  test mode:
	mode = bn_param['mode']	
	
	# 2. retrieve input dimension information and initialize accordingly
	num_data, dim_data = X.shape
	
	# -- 2.1 retrieve 'sequential' updated mean and variance, and perform initialization
	#        when it is the first iteration
	running_mean     = bn_param.get('running_mean', np.zeros(dim_data, dtype = X.dtype))
	running_variance = bn_param.get('running_variance', np.zeros(dim_data, dtype = X.dtype))
	
	# -- 2.2 initialize outputs and input_info tuple
	batchnorm_output, batchnorm_input_info = None, None
	
	# 3. Given mode (training or test), perform forward flow to evaluate the 
	# batch_norm output
	if mode == 'train':
		#######################################################################
        # Implement the training-time forward pass for batch norm.            #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
		
		# 3.1 calculate mean and variance of current batch of data
		sample_mean = np.mean(X, axis = 0)
		sample_variance = np.var(X, axis = 0)
		
		# 3.2 update 'sequential' updated mean and variance pool
		bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * sample_mean
		bn_param['running_variance'] = momentum * running_variance + (1 - momentum) * sample_variance
		
		# 3.3 calculate outputs from batch normalization
		input_normalized = (X - sample_mean) / (np.sqrt(sample_variance + eps))
		
		# 3.4 calculate outputs from batch normalization based on mean and variance from
		# previous iteration
		batchnorm_output = gamma * input_normalized + beta
		
		# 3.5 store input info as a tuple for being used in backprop run
		batchnorm_input_info = (X, gamma, beta, sample_mean, sample_variance, bn_param)
		
	elif mode == 'test':
		#######################################################################
        # Implement the test-time forward pass for batch normalization.       #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
		sample_mean = running_mean
		sample_variance = running_variance
		batchnorm_output = gamma * ((X - sample_mean) / (np.sqrt(sample_variance + eps))) \
							+ beta
		batchnorm_input_info = (X, gamma, beta, sample_mean, \
								 sample_variance, bn_param)
	else:
		raise ValueError('mode for performing batch normalization can not be empty')
	
	
	
	# 4. return outputs and input information tuple
	return batchnorm_output, batchnorm_input_info


##############################################################################
#
#
#	store all quantities towarding forward outputs
#
#
##############################################################################
def batchnorm_forward_allQuantities(X, gamma, beta, bn_param):
	'''
	Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
	'''
	# 1. read-in algorithmic settings from arguments
	eps = bn_param.get('eps', 1e-5) # -- numerical stabilizer
	momentum = bn_param.get('momentum', 0.9)
	
	# 1.1 we have two modes for performing batch normalization:
	# --  training mode:
	# --  test mode:
	mode = bn_param['mode']	
	
	# 2. retrieve input dimension information and initialize accordingly
	num_data, dim_data = X.shape
	
	# -- 2.1 retrieve 'sequential' updated mean and variance, and perform initialization
	#        when it is the first iteration
	running_mean     = bn_param.get('running_mean', np.zeros(dim_data, dtype = X.dtype))
	running_variance = bn_param.get('running_variance', np.zeros(dim_data, dtype = X.dtype))
	
	# -- 2.2 initialize outputs and input_info tuple
	batchnorm_output, batchnorm_input_info = None, None
	
	# 3. Given mode (training or test), perform forward flow to evaluate the 
	# batch_norm output
	if mode == 'train':
		#######################################################################
        # Implement the training-time forward pass for batch norm.            #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
		
		# 3.1 calculate mean (unit 1)
		#     X_mu: (D, )
		X_mu = 1 / float(num_data) * np.sum(X, axis = 0)
		
		# 3.2 substract X by its mean (unit 2)
		#     X_subst: (N, D)
		X_subst = X - X_mu
		
		# 3.3 square X_subst (unit 3)
		#     X_subst_sq: (N, D)
		X_subst_sq = X_subst ** 2
		
		# 3.4 calculate square of X substracted its mean (unit 4 -- variance)
		#     var: (D, )
		var = 1 / float(num_data) * np.sum(X_subst_sq, axis = 0)
		
		# 3.5 square root the variance to get std (unit 5 -- std)
		#     sqrt_var: (D, )
		sqrt_var = np.sqrt(var + eps)
		
		# 3.6 inverse the square root of variance  (unit 6 -- 1/std)
		#     invar: (D, 1)
		invar = 1 / sqrt_var
		
		# 3.7 multiply 1/std to mean substracted X (unit 7 -- normalized)
		#     X_hat: (N, D)
		X_hat = X_subst * invar
		
		# 3.8 multiply normalized X to gamma (unit 8)
		#     X_hat_gamma: (N, D)
		X_hat_gamma = gamma * X_hat
		
		# 3.9 batchnorm output: add beta to X_hat_gamma (unit 9)
		#     batchnorm_output: (N, D)
		batchnorm_output = X_hat_gamma + beta
		
		# 4. store all information incurred in this layer
		batchnorm_input_info = (X_mu, X_subst, X_subst_sq, var, sqrt_var, invar,\
								  X_hat, X_hat_gamma, X, gamma, beta, bn_param)
		
		# 5. update mean and variance of current batch normalization layer
		#    for being used in test mode
		running_mean = momentum * running_mean + (1 - momentum) * X_mu
		running_variance = momentum * running_variance + (1 - momentum) * var
		
		# 6. update the dictionary
		bn_param['running_mean'] = running_mean
		bn_param['running_variance'] = running_variance
		
	elif mode == 'test':
		#######################################################################
        # Implement the test-time forward pass for batch normalization.       #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
		sample_mean = running_mean
		sample_variance = running_variance
		X_hat = (X - sample_mean) / np.sqrt(sample_variance + eps)
		batchnorm_output = gamma * X_hat + beta
		batchnorm_input_info = (sample_mean, sample_variance, gamma, beta, bn_param)
	else:
		raise ValueError('mode for performing batch normalization can not be empty')
	
	
	
	# 4. return outputs and input information tuple
	return batchnorm_output, batchnorm_input_info

##### ------ implementation of batch normalization backprop ---- ####
def batchnorm_backward(dout, batchnorm_layer_input_info):
	'''
	Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	'''
	# 1. initialize gradients
	dX, dgamma, dbeta = None, None, None
	###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
	
	# 0. retrieve all input information fed to the batch normalization layer
	X_mu, X_subst, X_subst_sq, var, \
	sqrt_var, invar, X_hat, X_hat_gamma,\
	X, gamma, beta, bn_param = batchnorm_layer_input_info
	
	eps = bn_param.get('eps', 1e-5)
	N, D = X.shape
	
	# 1. start from the simplest gradients dbeta and dgamma
	dbeta = np.sum(dout, axis = 0)	
	dgamma = np.sum(X_hat * dout, axis = 0)
	
	# 2. backprop through 9 units until getting dloss/dx 
	
	# -- 2.1 derivative of unit 9 (dX_hat_gamma and dbeta)
	dX_hat_gamma = dout
	
	# -- 2.2 derivative of unit 8 (dX_hat and dgamma)
	dX_hat = gamma * dX_hat_gamma
	
	# -- 2.3 derivative of unit 7 (dX_subst_1 and dinvar)
	dX_subst_1 = invar * dX_hat
	
	# !!!!!!!!!!!!!!! dimension wrong !!!!!!!!!!
	# dinvar: (D,) which should be same as invar: (D, )
	#dinvar     = X_subst * dX_hat
	dinvar     = np.sum(X_subst * dX_hat, axis = 0)
	
	# -- 2.4 derivative of unit 6 (dsqrt_var)
	dsqrt_var = - 1 / (sqrt_var**2) * dinvar
	
	# -- 2.5 derivative of unit 5 (dvar)
	dvar = 1 / (2 * np.sqrt(var + eps)) * dsqrt_var
	
	# -- 2.6 derivative of unit 4 (dsq)
	dsq  = 1 / float(N) * dvar
	
	# -- 2.7 derivative of unit 3 (dX_subst_2)
	dX_subst_2 = 2 * X_subst * dsq
	
	# -- 2.8 derivative of unit 2 (dX_1 and dmu)
	dX_1 = (dX_subst_1 + dX_subst_2)
	
	## !!!!!! dimension wrong !!!!!! ##
	#  mu: (D,) --> dmu: (D, )
	#	dmu  = - (dX_subst_1 + dX_subst_2)	
	dmu   = - np.sum(dX_subst_1 + dX_subst_2, axis = 0)
	
	# -- 2.9 derivative of unit 1 (dX_2)
	dX_2 = 1 / float(N) * dmu
	
	# 3. gradient of X: dloss / dX
	dX   = dX_1 + dX_2
	
	return dX, dgamma, dbeta	
	



##### -------- compact implementation of batch normalization backprop -------####

def batchnorm_backward_fast(dout, batchnorm_layer_input_info):
	'''
	Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	'''
	# 1. initialize gradients
	dx, dgamma, dbeta = None, None, None
	###########################################################################
    # Implement the backward pass for batch normalization. Store the          #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
	
	# 2. retrieve batch normalization layer input info
	X, gamma, beta, sample_mean, sample_variance, bn_param = batchnorm_layer_input_info
	eps = bn_param['eps']
	# 3. we calculate gradients based on their difficulties
	# -- 3.1 dloss/dbeta = delta^{L+1} * dy/dbeta = delta^{L+1}
	dbeta = np.sum(dout, axis = 0)
	
	# -- 3.2 dloss/dgamaa = delta^{L+1} * dy/dgamma = delta^{L+1} * normalized_x
	dgamma = dout * ((X - sample_mean) / (np.sqrt(sample_variance + bn_param['eps'])))
	dgamma = np.sum(dgamma, axis = 0)
	
	# -- 3.3 dloss/dx = delta^{L+1} * dy/dx = delta^{L+1} * (gamma / std(x) - 2 * gamma / N)
	N = X.shape[0]
#	dydx = gamma/np.sqrt(sample_variance+eps) \
#		 - gamma/(N * np.sqrt(sample_variance+eps)) \
#		 + gamma / N**2 * (X - sample_mean)**2 * np.power(sample_variance,-1.5)\
#		 - gamma / N * (X - sample_mean)**2 * np.power(sample_variance, -1.5)
	dx = (1. / N) * gamma * (sample_variance + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)\
          - (X - sample_mean) * (sample_variance + eps)**(-1.0) * np.sum(dout * (X - sample_mean), axis=0))

	
	return dx, dgamma, dbeta





if __name__ == '__main__':
#	# Check the training-time forward pass by checking means and variances
#	# of features both before and after batch normalization
#	
#	# Simulate the forward pass for a two-layer network
#	np.random.seed(231)
#	N, D1, D2, D3 = 200, 50, 60, 3
#	X = np.random.randn(N, D1)
#	W1 = np.random.randn(D1, D2)
#	W2 = np.random.randn(D2, D3)
#	a = np.maximum(0, X.dot(W1)).dot(W2)
#	
#	print('Before batch normalization:')
#	print('  means: ', a.mean(axis=0))
#	print('  stds: ', a.std(axis=0))
#	
#	# Means should be close to zero and stds close to one
#	print('After batch normalization (gamma=1, beta=0)')
#	a_norm, _ = batchnorm_forward_allQuantities(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
#	print('  mean: ', a_norm.mean(axis=0))
#	print('  std: ', a_norm.std(axis=0))
#	
#	# Now means should be close to beta and stds close to gamma
#	gamma = np.asarray([1.0, 2.0, 3.0])
#	beta = np.asarray([11.0, 12.0, 13.0])
#	a_norm, _ = batchnorm_forward_allQuantities(a, gamma, beta, {'mode': 'train'})
#	print('After batch normalization (nontrivial gamma, beta)')
#	print('  means: ', a_norm.mean(axis=0))
#	print('  stds: ', a_norm.std(axis=0))
	
	
#	# Check the test-time forward pass by running the training-time
#	# forward pass many times to warm up the running averages, and then
#	# checking the means and variances of activations after a test-time
#	# forward pass.
#	np.random.seed(231)
#	N, D1, D2, D3 = 200, 50, 60, 3
#	W1 = np.random.randn(D1, D2)
#	W2 = np.random.randn(D2, D3)
#	
#	bn_param = {'mode': 'train'}
#	gamma = np.ones(D3)
#	beta = np.zeros(D3)
#	for t in range(50):
#	  X = np.random.randn(N, D1)
#	  a = np.maximum(0, X.dot(W1)).dot(W2)
#	  batchnorm_forward_allQuantities(a, gamma, beta, bn_param)
#	bn_param['mode'] = 'test'
#	X = np.random.randn(N, D1)
#	a = np.maximum(0, X.dot(W1)).dot(W2)
#	a_norm, _ = batchnorm_forward_allQuantities(a, gamma, beta, bn_param)
#	
#	# Means should be close to zero and stds close to one, but will be
#	# noisier than training-time forward passes.
#	print('After batch normalization (test-time):')
#	print('  means: ', a_norm.mean(axis=0))
#	print('  stds: ', a_norm.std(axis=0))
	
	
	# Gradient check batchnorm backward pass
	np.random.seed(231)
	N, D = 4, 5
	x = 5 * np.random.randn(N, D) + 12
	gamma = np.random.randn(D)
	beta = np.random.randn(D)
	dout = np.random.randn(N, D)
	
	bn_param = {'mode': 'train', 'eps':1e-5}
	fx = lambda x: batchnorm_forward_allQuantities(x, gamma, beta, bn_param)[0]
	fg = lambda a: batchnorm_forward_allQuantities(x, a, beta, bn_param)[0]
	fb = lambda b: batchnorm_forward_allQuantities(x, gamma, b, bn_param)[0]
	
	dx_num = eval_numerical_gradient_array(fx, x, dout)
	da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
	db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)
	
	_, cache = batchnorm_forward_allQuantities(x, gamma, beta, bn_param)
	dx, dgamma, dbeta = batchnorm_backward(dout, cache)
	print('dx error: ', rel_error(dx_num, dx))
	print('dgamma error: ', rel_error(da_num, dgamma))
	print('dbeta error: ', rel_error(db_num, dbeta))
	














































