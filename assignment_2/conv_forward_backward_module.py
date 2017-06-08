# -*- coding: utf-8 -*-
"""
Modules aim to implement forward/backward pass of convolutional networks

@author: xliu
"""
import numpy as np
from gradient_check import *
import matplotlib.pyplot as plt
from fast_layers import conv_forward_fast, conv_backward_fast
from time import time

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')

def conv_forward_naive(X, W, b, conv_param):
	'''
	A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
	'''
	
	###########################################################################
    # Implement the convolutional forward pass.                         	    #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
	
	# 1. retrieve input dimension information
	num_input, input_channel, input_height, input_width = X.shape
	
	# 2. retrieve filter dimension information
	num_filter, filter_channel, filter_height, filter_width = W.shape
	
	# 3. throw an exception if filter channel is different from input channel
	if filter_channel != input_channel:
		raise ValueError('filter and input should have identical channels.')
	
	# 4. retrieve filter spatial settings
	stride  = conv_param['stride']
	padding = conv_param['pad']
	
	# 5. padding input space to make conv-transformed volume 
	# has same dimension as original inputs:
	# -- we only pad the height and weight dimensions with #padding default valued as 0
	#    e.g X:np.ones(2, 3, 2, 2) with padding=1 and constant values = (4,5) 
	#    x[0,0,:,:]        = [1, 1;
	#	                      1, 1]
	# 
	# --> X_pad:(2, 3, 4, 4):
	#     X_pad[0,0,:,:]  = [4, 4, 4, 6;
	#	                     4, 1, 1, 6;
	#    	                 4, 1, 1, 6;
	#                       4, 6, 6, 6]
	X_pad = np.pad(X, ((0, ), (0, ), (padding, ), (padding, )), 'constant')
	
	
	# 6. calculate the height/width of transformed local region
	activationMap_height = int((input_height - filter_height + 2 * padding) / stride + 1)
	activationMap_width  = int((input_width - filter_width + 2 * padding) / stride + 1)
	
	# 7. initialize output variable
	conv_out = np.zeros((num_input, num_filter, activationMap_height, activationMap_width))
	
	# 8. naively loop over all local region of input space
	
	# -- 8.1 loop over all input points (images)
	for n in range(num_input):
		# -- 8.2 loop over all filters
		for f in range(num_filter):
			# -- 8.3 loop over all heights/rows in receptive regions
			for h in range(activationMap_height):
				# -- 8.4 loop over all widths/colums in receptive regions
				for w in range(activationMap_width):
					# weights are different among filters, but within a filter
					# weight is same for being applied to local regions, so-called 
					# parameter sharing.
					dot_prod_localXpad_weight = X_pad[n, :, \
												  h*stride: h*stride+filter_height,\
												  w*stride: w*stride+filter_width] * W[f,:]
					
					# sum-up all results from element-wise matrix multiplication, plue bias
					conv_out[n, f, h, w] = np.sum(dot_prod_localXpad_weight) + b[f]
	
	
	# 9 . cache all input information to create a tuple
	conv_input_info = (X, W, b, conv_param)
	
	# 10. return outputs and input_info
	return conv_out, conv_input_info
	

# convolution naive backward pass
def conv_backward_naive(dout, conv_layer_input_info):
	'''
	A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
	'''
	# 1. retrieve input information from tuple
	X, W, b, conv_param = conv_layer_input_info
	stride = conv_param['stride']
	
	# 2. padding raw X if necessary
	padding = conv_param['pad']
	X_pad = np.pad(X, ((0, ), (0, ), (padding, ), (padding, )), 'constant')
	
	# 3. retrieve dimension information
	num_input, channel_input, height_input, width_input = X.shape
	num_filter, channel_filter, height_filter, width_filter = W.shape
	
	# num_filter_output == num_filter
	num_output, num_filter_output, height_output, width_output = dout.shape
	
	# -- 3.1 exception detection
	if channel_input != channel_filter:
		raise ValueError('channel of inputs have to be identical to channel of each filter')
	
	# -- 3.2 retrieve dimension of activation maps
	num_activationMap_regions, depth_activationMap_regions, height_activationMap, width_activationMap = dout.shape
	if num_activationMap_regions != num_input:
		raise ValueError('number of activation maps has to be identical number of inputs/images')
	if depth_activationMap_regions != num_filter:
		raise ValueError('depth of activation maps has to be identical number of filters')
	
	
	# 4. naively loop over all layers to get dloss/db
	db = np.zeros((num_filter))
	for filter_idx in range(num_filter):
		db[filter_idx] = np.sum(dout[:, filter_idx, :, :])
	
	# 5. naively loop over all layers to get dloss/dw
	dw = np.zeros((num_filter, channel_filter, height_filter, width_filter))
	
	for filter_idx in range(num_filter):
		for channel_idx in range(channel_filter):
			for i in range(height_filter):
				for j in range(width_filter):
					X_pad_subset = X_pad[:, channel_idx, \
					i:(i+height_output*stride):stride, \
					j:(j+width_output*stride):stride]
					
					dw[filter_idx, channel_idx, i, j] = np.sum(dout[:, filter_idx, :, :] * X_pad_subset)
	
	
	# 6. naively loop over all layers to get dloss/dx
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#
	# dloss/dx = dloss/dxpad * dxpad/dx
	#
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	dx = np.zeros((num_input, channel_input, height_input, width_input))
	
	# 9 loops!
	for n in range(num_input): # input data points
		for c in range(channel_input): # channels in each input point
			for height_input_idx in range(height_input): # row in a input point
				for width_input_idx in range(width_input): # col in a input point
					for num_filter_idx in range(num_filter): # filters in convnet
						for height_output_idx in range(height_output): # row in an output point
							for width_output_idx in range(width_output): # col in an output point
								for height_filter_idx in range(height_filter): # row in each filter
									for width_filter_idx in range(width_filter): # col in each filter
										# ------------------------------------
										#
										# see if the current x(data_idx, 
										#	                   channel_idx, 
										# 	                   input_height_idx, 
										#                      input_width_idx)
										# is a zero padded
										#
										# -------------------------------------
										if (height_filter_idx + height_output_idx * stride == height_input_idx + padding) \
										& (width_filter_idx  + width_output_idx * stride  == width_input_idx  + padding):
											dx[n, c, height_input_idx, width_input_idx] \
											+= dout[n, num_filter_idx, height_output_idx, width_output_idx] \
											* W[num_filter_idx, c, height_filter_idx, width_filter_idx] 
														
						
		
	
	return dx, dw, db


###### max pooling forward layer
def max_pool_forward_naive(X, pool_param):
	'''
	A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
	'''
	# 1. retrieve pool hyper-parameters
	stride = pool_param['stride']
	pool_height = pool_param['pool_height']
	pool_width  = pool_param['pool_width']
	
	# 2. retrieve input dimension information
	num_input, channel_input, height_input, width_input = X.shape
	
	# 3. evaluate the dimension of output
	num_pool_out   = num_input
	depth_pool_out = channel_input
	height_pool_out = int((height_input - pool_height) / stride + 1)
	width_pool_out  = int((width_input - pool_width) / stride + 1)
	
	# 4. initialize output matrix
	maxpool_out = np.zeros((num_pool_out, depth_pool_out, height_pool_out, width_pool_out))	
	
	# 5. naively loop over volumes
	for n in range(num_input):
		for c in range(channel_input):
			for h in range(height_pool_out):
				for w in range(width_pool_out):
					maxpool_out[n, c, h, w] = np.max(X[n, c,\
														h*stride:h*stride+pool_height,\
														w*stride:w*stride+pool_width])
	# 6. store maxpool layer input information
	maxpool_input_info = (X, pool_param)
	
	return maxpool_out, maxpool_input_info
	
### maxpooling backward naive implementation
def max_pool_backward_naive(dout, maxpool_input_info):
	'''
	A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
	'''
	# 1. unpack input information of maxpooling layer
	X, pool_param = maxpool_input_info
	stride = pool_param['stride']
	pool_height = pool_param['pool_height']
	pool_width  = pool_param['pool_width']
	
	# 2. retrieve dimension information from tuple encoded
	num_input, channel_input, height_input, width_input = X.shape
	
	# 3. evaluate height and width of output after maxpooling
	height_out = int((height_input - pool_height)/stride + 1)
	width_out = int((width_input - pool_width)/stride + 1)
	
	# 4. create a gradient matrix
	dx = np.zeros((num_input, channel_input, height_input, width_input))	
	
	# 5. naively loop over all points and layers
	# -- key point: create a mask to indicate which variable in local region
	#    is gone through the maxpool, then dout will flow through accordingly
	for n in range(num_input):
		for c in range(channel_input):
			for h in range(height_out):
				for w in range(width_out):
					# 5.1 retrieve the local region that maxpool filter will slide through
					x_local_region = X[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]
					# 5.2 get the activation after sliding by maxpool filter
					maxpool_activation = np.max(x_local_region)
					# 5.3 create a mask (spatial information) to indicate which variable in
					#     local region has gone through
					mask = 1 * (x_local_region == maxpool_activation)
					# 5.4 dot product the dout and mask to get grad that successfully flow
					#     through.
					dx[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]\
					+= dout[n, c, h, w] * mask
	
	return dx
	
############ --- spatial batch normalization forward --- ######################
def spatial_batchnorm_forward(X, gamma, beta, bn_param):
	'''
	Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
		training, we need to update running_mean/running_variance using momentum
		test, we set sample_mean/sample_variance to running_mean/running_variance without updated
		with momentum.
		
		training:
		running_mean = momentum * running_mean + (1-momentum) * sample_mean
		running_variance = momentum * running_variance + (1-momentum) * sample_variance
		running_mean(s)/running_variance(s) are stored in layer individually dictionaries.
		Those quantities will be updated along mini-batch forward/backward iterations.
		
		test: 
		X_hat = (X-running_mean) / (sqrt(running_variance))
		X_bn_out = gamma * X_hat + beta
		
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
	'''
	# bn_param: eps, momentum, running_mean, running_variance
	
	# 1. retrieve hyper-parameters
	eps = bn_param.get('eps', 1e-5)
	mode = bn_param['mode']
	momentum = bn_param.get('momentum', 0.9)
	running_mean = bn_param['running_mean']
	running_variance = bn_param['running_variance']
	
	# 2. update formula:
	# -- 2.1 running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	# -- 2.2 running_variance = momentum * running_variance + (1 - momentum) * sample_variance
	# -- 2.3 bn_out = gamma * X_hat + beta
	# at the end of day, mean(bn_out, axis=0) == beta, which is initialized by user
	# at the end of day, std(bn_out, axis=0) == gamma, which is initialized by user
	
	return spatial_bn_out, spatial_bn_input_info
	
##############################################################################
#
#	self-test code
#
##############################################################################
if __name__ == '__main__':
	
	test_mode = 'conv_fast_version'
	
	if test_mode == 'conv_naive_forward':
		# -- test convolution naive forward pass -- #
		x_shape = (2, 3, 4, 4)
		w_shape = (3, 3, 4, 4)
		x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
		w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
		b = np.linspace(-0.1, 0.2, num=3)
		
		conv_param = {'stride': 2, 'pad': 1}
		out, _ = conv_forward_naive(x, w, b, conv_param)
		correct_out = np.array([[[[-0.08759809, -0.10987781],
		                           [-0.18387192, -0.2109216 ]],
		                          [[ 0.21027089,  0.21661097],
		                           [ 0.22847626,  0.23004637]],
		                          [[ 0.50813986,  0.54309974],
		                           [ 0.64082444,  0.67101435]]],
		                         [[[-0.98053589, -1.03143541],
		                           [-1.19128892, -1.24695841]],
		                          [[ 0.69108355,  0.66880383],
		                           [ 0.59480972,  0.56776003]],
		                          [[ 2.36270298,  2.36904306],
		                           [ 2.38090835,  2.38247847]]]])
		
		# Compare your output to ours; difference should be around 2e-8
		print('Testing conv_forward_naive')
		print('difference: ', rel_error(out, correct_out))
		
	elif test_mode == 'conv_naive_forward_kitten_puppy':
		# read-in kitten_puppy data:
		# run loadData()
		
		# Set up a convolutional weights holding 2 filters, each 3x3
		w = np.zeros((2, 3, 3, 3))
		
		# num_filter, filter_channel, filter_height, filter_width = W.shape
		# filter_1: w[0, :, :, :] -- detect grayscale
		# filter_2: w[1, :, :, :] -- detect edge
		
		# The first filter converts the image to grayscale.
		# Set up the red, green, and blue channels of the filter.
		w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
		w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
		w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
		
		# Second filter detects horizontal edges in the blue channel.
		# -- w[0, 0:1, :, :] = 0
		w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
		
		# Vector of biases. We don't need any bias for the grayscale
		# filter, but for the edge detection filter we want to add 128
		# to each output so that nothing is negative.
		b = np.array([0, 128])
		
		
		# Compute the result of convolving each input in x with each filter in w,
		# offsetting by b, and storing the results in out.
		out, _ = conv_forward_naive(kitten_puppy_stacked_data, w, b, {'stride': 1, 'pad': 1})
		
		# Show the original images and the results of the conv operation
		plt.subplot(2, 3, 1)
		imshow_noax(puppy, normalize=False)
		plt.title('Original image')
		
		# grayscale is abstracted by w[0, 0:2, :, :]
		plt.subplot(2, 3, 2)
		imshow_noax(out[0, 0])
		plt.title('Grayscale')
		
		# edge is abstracted by w[1, 2, :, :]
		plt.subplot(2, 3, 3)
		imshow_noax(out[0, 1])
		plt.title('Edges')
		
		plt.subplot(2, 3, 4)
		imshow_noax(kitten_cropped, normalize=False)
		
		# grayscale is abstracted by w[0, 0:2, :, :]
		plt.subplot(2, 3, 5)
		imshow_noax(out[1, 0])
		
		# edge is abstracted by w[1, 2, :, :]
		plt.subplot(2, 3, 6)
		imshow_noax(out[1, 1])
		plt.show()
	
	elif test_mode == 'conv_naive_backward':
		# -- test convolution naive forward pass -- #
		np.random.seed(231)
		x = np.random.randn(4, 3, 5, 5)
		w = np.random.randn(2, 3, 3, 3)
		b = np.random.randn(2,)
		dout = np.random.randn(4, 2, 5, 5)
		conv_param = {'stride': 1, 'pad': 1}
		
		dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
		dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
		db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
		
		out, cache = conv_forward_naive(x, w, b, conv_param)
		print('conv_naive_backprop start')
		dx, dw, db = conv_backward_naive(dout, cache)
		print('conv_naive_backprop finish')
		# Your errors should be around 1e-8'
		print('Testing conv_backward_naive function')
		print('dx error: ', rel_error(dx, dx_num))
		print('dw error: ', rel_error(dw, dw_num))
		print('db error: ', rel_error(db, db_num))
	
	elif test_mode == 'conv_fast_version':
		np.random.seed(231)
		# num_input, channel_input, height_input, width_input
		x = np.random.randn(10, 3, 5, 5)
		
		# num_filter, channel_filter, height_filter, width_filter
		# -- channel_input == channel_filter
		# conv filter hyper-parameter: (num_filter, height_filter, width_filter, stride, pad)
		# maxpool filter hyper-parameter: (height_maxpool_filter, width_maxpool_filter, stride)
		w = np.random.randn(4, 3, 3, 3)
		
		# b: (num_filter)
		b = np.random.randn(4,)
		
		# dout: (num_input, num_filter, height_activation, width_activation)
		# -- height_activation = (height_input - height_filter + 2 * padd) /2 -1
		# -- width_activation = (width_input - width_filter + 2 * padd) /2 -1
		dout = np.random.randn(10, 4, 3, 3)
		conv_param = {'stride': 2, 'pad': 1}
		
		t0 = time()
		out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
		t1 = time()
		out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
		t2 = time()
		
		print('Testing conv_forward_fast:')
		print('Naive: %fs' % (t1 - t0))
		print('Fast: %fs' % (t2 - t1))
		print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
		print('Difference: ', rel_error(out_naive, out_fast))
		
		t0 = time()
		dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
		t1 = time()
		dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
		t2 = time()
		
		print('\nTesting conv_backward_fast:')
		print('Naive: %fs' % (t1 - t0))
		print('Fast: %fs' % (t2 - t1))
		print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
		print('dx difference: ', rel_error(dx_naive, dx_fast))
		print('dw difference: ', rel_error(dw_naive, dw_fast))
		print('db difference: ', rel_error(db_naive, db_fast))
	
	elif test_mode == 'max_pool_forward_naive':
		x_shape = (2, 3, 4, 4)
		x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
		pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
		
		out, _ = max_pool_forward_naive(x, pool_param)
		
		correct_out = np.array([[[[-0.26315789, -0.24842105],
		                          [-0.20421053, -0.18947368]],
		                         [[-0.14526316, -0.13052632],
		                          [-0.08631579, -0.07157895]],
		                         [[-0.02736842, -0.01263158],
		                          [ 0.03157895,  0.04631579]]],
		                        [[[ 0.09052632,  0.10526316],
		                          [ 0.14947368,  0.16421053]],
		                         [[ 0.20842105,  0.22315789],
		                          [ 0.26736842,  0.28210526]],
		                         [[ 0.32631579,  0.34105263],
		                          [ 0.38526316,  0.4       ]]]])
		
		# Compare your output with ours. Difference should be around 1e-8.
		print('Testing max_pool_forward_naive function:')
		print('difference: ', rel_error(out, correct_out))

	elif test_mode == 'max_pool_backward_naive':
		np.random.seed(231)
		x = np.random.randn(3, 2, 8, 8)
		dout = np.random.randn(3, 2, 4, 4)
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
		
		dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
		
		out, cache = max_pool_forward_naive(x, pool_param)
		dx = max_pool_backward_naive(dout, cache)
		
		# Your error should be around 1e-12
		print('Testing max_pool_backward_naive function:')
		print('dx error: ', rel_error(dx, dx_num))


































