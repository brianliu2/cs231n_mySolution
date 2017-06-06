# -*- coding: utf-8 -*-
"""
Different types of gradient descent:

vanilla stochastic gradient descent (SGD) as our update rule. 
More sophisticated update rules can make it easier to train deep networks. 
We will implement a few of the most commonly used update rules and compare them
to vanilla SGD.

1: SGD+Momentum
2: RMSProp
3: Adam

@author: xliu
"""

import numpy as np
from solver import * 
from class_fullyConnectedNet import *

def rel_error(x, y):
    '''returns relative error'''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

'''
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
def update(w, dw, config=None):
Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.
For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
'''

def sgd(w, dw, config=None):
	"""
	Performs vanilla stochastic gradient descent.
	config format:
	- learning_rate: Scalar learning rate.
	"""
	if config is None: 
		config = {}
	config.setdefault('learning_rate', 1e-2)

	w -= config['learning_rate'] * dw
	return w, config

def sgd_momentum(w, dw, config=None):
	'''
	Stochastic gradient descent with momentum is a widely used update rule that
	tends to make deep networks converge faster than vanilla stochstic gradient 
	descent.
	
	The SGD+momentum update rule in the function sgd_momentum and run the testing
	code to check the implementation. One should see errors less than 1e-8.
	
	Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients. (Here, w is just a symbol of variable,
	  it can be model input x, or model input weights w, or model bias b)
	
	formula (similar to annealing cooling schedule):
	velocity = mu * velocity - learn_rate * gradient
	x        = x + velocity
	'''
	# 1. if there is no optim_config, then we initialize as an empty dictionary
	if config is None:
		config = {}
	
	# 2. set the default values if necessary
	config.setdefault('learning_rate', 1e-3)
	learning_rate = config['learning_rate']
	
	# 2.1 momentum is the mu in formula, where it can be interpreted as cooling
	# factor. The common choice could be 0.5, 0.9 and 0.99
	config.setdefault('momentum', 0.9)
	momentum = config['momentum']
	
	# 2.1. initialize the velocity of w
	velocity = config.get('velocity', np.zeros_like(w))
	
	# 3. update velocity of w and correspondingly the 'position' of w
	velocity = momentum * velocity - learning_rate * dw
	next_w = w + velocity 
	
	# 4. update the velocity values in dictionary
	config['velocity'] = velocity
	
	return next_w, config






###########################################################################################
#
#
#
#	Implementation testing
#
#
#
###########################################################################################

if __name__ == '__main__':
	N, D = 4, 5
	w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
	dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
	v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
	config = {'learning_rate': 1e-3, 'velocity': v}
	next_w, _ = sgd_momentum(w, dw, config=config)

	expected_next_w = np.asarray([
	[ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
	[ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
	[ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
	[ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
	
	expected_velocity = np.asarray([
	[ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
	[ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
	[ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
	[ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])
	
	print('next_w error: ', rel_error(next_w, expected_next_w))
	print('velocity error: ', rel_error(expected_velocity, config['velocity']))


	'''
	Once you have done so, run the following to train a six-layer network with 
	both SGD and SGD+momentum. You should see the SGD+momentum update rule converge faster.
	'''
	num_train = 50
	small_data = {
	  'X_train': data['X_train'][:num_train],
	  'y_train': data['y_train'][:num_train],
	  'X_val': data['X_val'],
	  'y_val': data['y_val'],
	}
	
	solvers = {}
	
	for update_rule in ['sgd', 'sgd_momentum']:
		print('running with ', update_rule)
		model = fullyConnectedNet([100, 100, 100, 100, 100], input_dims = 32 * 32 * 3, \
									 num_classes = 10, weight_scale=5e-2)
		
		Solver = solver(model, small_data, num_epochs=5, batch_size=100, 
						 update_rule=update_rule, opt_config = {'learning_rate':1e-2},
						 verbose=True)
		Solver.train()
		solvers[update_rule] = Solver
		print('training finished.')
		
		


  
  
  
  

  
  
  
  






















































































