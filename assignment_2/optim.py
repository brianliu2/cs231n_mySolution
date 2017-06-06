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

def Nesterov_Accelerated_Gradient(w, dw, config=None):
	'''
	This is a one-step ahead of sgd_momentum, it calculates the gradient 
	along the momentum direction, rather than the current x. For the sake of
	computational convenience, we perform a variable tranformation
	
	formula:
	1. phi_{t-1} = w_{t-1} + mu * velocity_{t-1} --> variable transform
	2. velocity_{t} = mu * velocity_{t-1} - learning_rate * dPhi_{t-1}
	3. phi_{t} = phi_{t-1} - mu * velocity_{t-1} + (1+mu) * velocity_{t}
	4. phi_{0} = w_{0}
	5. w_{t} = phi_{t} - mu * velocity_{t-1}
	         = phi_{t-1} - mu * velocity_{t-1} + (1+mu) * velocity_{t} - mu * velocity_{t-1}
			  = 
	'''
	# 1. initialize if config is None
	if config is None:
		config = {}
	
	# 2. set default values to hyper-parameters
	config.setdefault('momentum', 0.9)
	config.setdefault('learning_rate', 1e-3)
	
	momentum = config['momentum']
	learning_rate = config['learning_rate']
	
	velocity = config.get('velocity', np.zeros_like(w))
	
	# 3. calculate the transformed variable -- phi
	velocity_prev = velocity
	velocity = momentum * velocity - learning_rate * dw
	next_w = w - momentum * velocity_prev + (1+momentum) * velocity
	
	# 4. update config dictionary
	config['velocity'] = velocity
	
	return next_w, config
	
def rmsprop(w, dw, config=None):
	'''
	Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
				
	Intuition: at the steep gradient (dw is large) feature direction, we make smaller move at
	every descent, at the shallow gradient (dw is small) feature direction, we make a 
	relatively large move. However, as forward-backprop goes along, gradients tend to be 
	smaller and smaller, therefore, as the denominator, it will become larger and 
	larger, consequently, w is not going anywhere. This is called AdaGrad.
	
	As a step advanced, rmsprop, we consider a decay rate for the cumulation of
	gradient in a step past (this is called leak of gradient info), in associated with 
	information about whether the current feature has relatively large/small gradient.
	If it is large, make a tiny step, while it is small, make a larger step.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
				
	formula:
	(adagrad): cache = dw * dw
	(rmspro) : cache_{t+1} = decay * (cache_{t}) + (1- decay) * (dw * dw)
	w = w - learning_rate * dw / (sqrt(cache) + 1e-7)
	'''
	# 1. initialize if config is None
	if config is None:
		config = {}
	
	# 2. set default values to hyper-parameters
	config.setdefault('decay_rate', 0.99)
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('epsilon', 1e-8)
	
	decay_rate = config['decay_rate']
	learning_rate = config['learning_rate']
	epsilon = config['epsilon']
	cache = config.get('cache', np.zeros_like(dw))
	
	
	# 3. calculate leak of gradient accumulation and large/small scale of gradients
	cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
		
	next_w = w - (learning_rate * dw) / (np.sqrt(cache) + epsilon)
	
	# 4. update config dictionary
	config['cache'] = cache
	
	return next_w, config


def adam(w, dw, config = None):
	'''
	Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - iteration: Iteration number.
				
	formula: (intuition, we not only consider the direction of gradient, 
			   but also how larger/small they are)
	
	1. momentum = decay_rate_1 * momentum + (1 - decay_rate_1) * dw
	2. velocity = decay_rate_2 * velocity + (1 - decay_rate_2) * (dw * dw)
	
	3 . w = w - learning_rate * momentum / (sqrt(velocity) + 1e-7)
	'''
	# 1. initialize config if necessary
	if config is None:
		config = {}
		
	# 2. set default values to hyper-parameters if necessary
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('decay_rate_moment', 0.9)
	config.setdefault('decay_rate_velocity', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('iteration', 0)
	#config.setdefault('shrink_considered_iteration', 5)
	
	learning_rate = config['learning_rate']
	decay_rate_moment = config['decay_rate_moment']
	decay_rate_velocity = config['decay_rate_velocity']
	epsilon = config['epsilon']
	#shrink_considered_iteration = config['shrink_considered_iteration']
	iteration = config['iteration']
	
	momentum = config.get('momentum', np.zeros_like(w))
	velocity = config.get('velocity', np.zeros_like(w))
	
	# 3. update the w based on formula
	iteration += 1
	momentum = decay_rate_moment * momentum + (1 - decay_rate_moment) * dw
	velocity = decay_rate_velocity * velocity + (1 - decay_rate_velocity) * (dw ** 2)
	
	
	
	# 3.1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#
	#	Don't forget to count the iteration number: adaptive shrink the momentum and velocity
	# 	as the iteraion rolls
	#	shrink_considered_iteration: big_number
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
#	for t in range(1, shrink_considered_iteration+1):
#		momentum_hat = momentum / (1 - (decay_rate_moment)**t)
#		velocity_hat = velocity / (1 - (decay_rate_velocity)**t)
#		next_w = (w - learning_rate * momentum_hat) / (np.sqrt(velocity_hat) + epsilon)
#	else:
#		next_w = (w - learning_rate * momentum) / (np.sqrt(velocity) + epsilon)
	
	# momentum' = momentum / (iteration-related discount): this is to enlarge the 
	# momentum at the first few iterations, because it is initialized as zero.
	# so it will be relatively smaller than what it should actually be. This is because
	# as long as iteration becomes larger and larger, the denominator (1 - (decay_rate_moment)**iteration)
	# close to 1. The same reason applies to velocity.
	
	momentum_hat = momentum / (1 - (decay_rate_moment)**iteration)
	velocity_hat = velocity / (1 - (decay_rate_velocity)**iteration)
	next_w = w - learning_rate * momentum_hat / (np.sqrt(velocity_hat + epsilon))
	
	config['momentum'] = momentum
	config['velocity'] = velocity
	config['iteration'] = iteration
	
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
	m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
	v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)
	
	#config = {'learning_rate': 1e-2, 'momentum': m, 'velocity': v, 'iteration': 5, 'shrink_considered_iteration':30}
	config = {'learning_rate': 1e-2, 'momentum': m, 'velocity': v, 'iteration': 5}
	next_w, _ = adam(w, dw, config=config)
	
	expected_next_w = np.asarray([
	  [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
	  [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
	  [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
	  [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
	expected_v = np.asarray([
	  [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
	  [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
	  [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
	  [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
	expected_m = np.asarray([
	  [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
	  [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
	  [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
	  [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])
	
	print('next_w error: ', rel_error(expected_next_w, next_w))
	print('v error: ', rel_error(expected_v, config['velocity']))
	print('m error: ', rel_error(expected_m, config['momentum']))


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
	
	for update_rule in ['sgd', 'rmsprop']:
		print('running with ', update_rule)
		model = fullyConnectedNet([100, 100, 100, 100, 100], input_dims = 32 * 32 * 3, \
									 num_classes = 10, weight_scale=5e-2)
		
		Solver = solver(model, small_data, num_epochs=15, batch_size=100, 
						 update_rule=update_rule, opt_config = {'learning_rate':1e-2},
						 verbose=True)
		Solver.train()
		solvers[update_rule] = Solver
		print('training finished.')
		
	# print-out results
	plt.subplot(3, 1, 1)
	plt.title('Training loss')
	plt.xlabel('Iteration')
	
	plt.subplot(3, 1, 2)
	plt.title('Training error')
	plt.xlabel('Epoch')
	
	plt.subplot(3, 1, 3)
	plt.title('Validation error')
	plt.xlabel('Epoch')
	
	for update_rule, solver in list(solvers.items()):
		plt.subplot(3, 1, 1)
		plt.plot(solver.loss_history, 'o', label=update_rule)
		
		plt.subplot(3, 1, 2)
		plt.plot(solver.train_err_history, '-o', label=update_rule)
		
		plt.subplot(3, 1, 3)
		plt.plot(solver.valid_err_history, '-o', label=update_rule)
		
	for i in [1, 2, 3]:
		plt.subplot(3, 1, i)
		plt.legend(loc='upper center', ncol=4)
	
	plt.gcf().set_size_inches(15, 15)
	plt.show()
		
		


  
  
  
  

  
  
  
  






















































































