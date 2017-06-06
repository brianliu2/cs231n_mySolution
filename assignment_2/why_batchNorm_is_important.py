# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:20:40 2017

@author: xliu
"""
import numpy as np
import matplotlib.pyplot as plt

D = np.random.randn(1000, 500)
hidden_layers_size = [500] * 10

# choose a nonlinear function to be the activation function
nonlinearities = ['relu'] * len(hidden_layers_size)

act = {'relu': lambda x: np.maximum(0, x), 'tanh': lambda x: np.tanh(x)}
Hs  = {}

for i in range(len(hidden_layers_size)):
	if i == 0:
		X = D
	else:
		X = Hs[i-1]
	
	fan_in = X.shape[1]
	fan_out = hidden_layers_size[i]
	# 1. naive tiny value initialization
#	W = 0.01 * np.random.randn(fan_in, fan_out)
	
	# 2. Xavier weight initialization: force points to have variance value to be one
	# -- disadvantage: doesn't account nonliearity of activation: 
	# e.g. tanh in this example has relatively week nonlinearity.
	# if we change the activation to ReLu, we basically kill half of points to zero,
	# which means we half the variance
#	W = (1/np.sqrt(fan_in)) * np.random.randn(fan_in, fan_out)
	
	# 3. He weight initialization: after ReLu, half points are killed, as a result
	# of which, variance is manually halved. So in the weight initialization, we 
	# manually double its variance. We would saturate outputs from hidden layers.
	W = (1/(np.sqrt(fan_in/2))) * np.random.randn(fan_in, fan_out)
	
	H = np.dot(X, W)
	H = act[nonlinearities[i]](H)
	Hs[i] = H
	
# look at distributions at each layer
print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))
layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds  = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
	print('hidden layer %d had mean %f and std %f' % (i+1, layer_means[i], layer_stds[i]))
	
# plot means and standard deviations
keys = [key for key in Hs.keys()]
plt.figure()
plt.subplot(121)
plt.plot(keys, layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(keys, layer_stds, 'or-')
plt.title('layer std')

# plot the raw distributions
plt.figure()
for i, H in Hs.items():
	plt.subplot(1, len(Hs), i+1)
	plt.hist(H.ravel(), 30, range=(-1,1))

















































