# -*- coding: utf-8 -*-
"""
Solver
In the previous assignment, the logic for training models was 
coupled to the models themselves. Following a more modular design, 
for this assignment we have split the logic for training models 
into a separate class.

@author: xliu
"""

import numpy as np
import optim
from class_twoLayerModularNN import *

class solver(object):
	'''
	A Solver encapsulates all the logic necessary for training classification
	models. The Solver performs stochastic gradient descent using different
	update rules defined in optim.py.
	The solver accepts both training and validataion data and labels so it can
	periodically check classification accuracy on both training and validation
	data to watch out for overfitting.
	To train a model, you will first construct a Solver instance, passing the
	model, dataset, and various optoins (learning rate, batch size, etc) to the
	constructor. You will then call the train() method to run the optimization
	procedure and train the model.
	
	After the train() method returns, model.params will contain the parameters
	that performed best on the validation set over the course of training.
	In addition, the instance variable solver.loss_history will contain a list
	of all losses encountered during training and the instance variables
	solver.train_acc_history and solver.val_acc_history will be lists containing
	the accuracies of the model on the training and validation set at each epoch.
	
	Example usage might look something like this:
	
	data = {
	'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
			}
	model = MyAwesomeModel(hidden_size=100, reg=10)
	solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
	solver.train()
	A Solver works on a model object that must conform to the following API:
	- model.params must be a dictionary mapping string parameter names to numpy
	arrays containing parameter values.
	- model.loss(X, y) must be a function that computes training-time loss and
	gradients, and test-time classification scores, with the following inputs
    and outputs:
	
    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].
    
	Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].
    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
	'''

	def __init__(self, model, data, **kwargs):
		'''
	    Construct a new Solver instance.
	    
	    Required arguments:
	    - model: A model object conforming to the API described above
	    - data: A dictionary of training and validation data with the following:
	      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
	      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
	      'y_train': Array of shape (N_train,) giving labels for training images
	      'y_val': Array of shape (N_val,) giving labels for validation images
	      
	    Optional arguments:
	    - update_rule: A string giving the name of an update rule in optim.py.
	      Default is 'sgd'.
	    - optim_config: A dictionary containing hyperparameters that will be
	      passed to the chosen update rule. Each update rule requires different
	      hyperparameters (see optim.py) but all update rules require a
	      'learning_rate' parameter so that should always be present.
	    - lr_decay: A scalar for learning rate decay; after each epoch the learning
	      rate is multiplied by this value.
	    - batch_size: Size of minibatches used to compute loss and gradient during
	      training.
	    - num_epochs: The number of epochs to run for during training.
	    - print_every: Integer; training losses will be printed every print_every
	      iterations.
	    - verbose: Boolean; if set to false then no output will be printed during
	      training.
		'''
		# 1. load-in data and model class
		self.model = model
		self.X_train = data['X_train']
		self.y_train = data['y_train']
		self.X_valid = data['X_val']
		self.y_valid = data['y_val']
		
		# 2. unpack keyword arguments 
		
		# 2.1 -- kwargs.get(kw, default_val): set default_val to kw
		self.update_rule = kwargs.get('update_rule', 'sgd')
		
		# 2.2 -- optimization configuration:
		# 	     e.g. hyper-parameters for momentum-sgd, adam
		self.optim_config = kwargs.pop('optim_config', {})
		
		# 2.3 -- set the learn_rate decay factor to 0.95, i.e. learn rate of optimization
		#        is decay at each epoch (itearion_per_epoch = num_train / batch_size)
		self.lr_decay = kwargs.get('lr_decay', 0.95)
		
		# 2.4 -- self-explanatory keywords
		self.batch_size = kwargs.get('batch_size', 100)
		self.num_epochs = kwargs.get('num_epochs', 10)
		
		# 2.5 -- status print-out keywords
		self.print_every = kwargs.get('print_every', 10)
		self.verbose = kwargs.get('verbose', True)
		
		# 2.6 -- throw an error if there are unwanted/extra keyword arguments
#		if len(kwargs) > 0:
#			extra = ', '.join('"%s"' % kw for kw in kwargs.keys())
#			raise ValueError('Unrecognized arguments %s' % extra)
			
		# 2.7 -- update_rule -- optimization method can not be empty
		#        hasattr(object, attribute) -- check whether object has attribute
		if not hasattr(optim, self.update_rule):
			raise ValueError('Invalid update_rule (optimization) "%s"' % self.update_rule)
		
		# 2.8 -- if the update_rule exists, then map the string name (self.update_rule)
		#        into an actual function, because self.update_rule is a string type variable
		#         when it is read-in
		self.update_rule = getattr(optim, self.update_rule)
		
		# 3. create 'booking-keeping' variables in staying train information tracked
		self._createTrackInfo()
	
	### -------------------------------------------------------- ####
	#                                                                  
	#	a function to create book-keeping variables for optimization  
	#   this function can't be called separated.      	               
	#                                                                 
	### -------------------------------------------------------- ####
	def _createTrackInfo(self):
		# 1. epoch counter
		self.current_epoch = 0
		self.best_valid_err = np.inf
		self.best_val_acc = 0
		self.best_params = {}
		self.loss_history = []
		self.train_err_history = []
		self.valid_err_history = []
		
		# 2. make a deep copy of the optim_config for each parameter of neural network
		# class, e.g. W1, b1, W2, b2,......
		
		self.optim_nn_params_configs = {}
		for nn_para in self.model.params:
			dict_nn_para_opt_config = {kw: val for kw, val in self.optim_config.items()}
			self.optim_nn_params_configs[nn_para] = dict_nn_para_opt_config
	
	
	### -------------------------------------------------------- ####
	#        
	#   modularized train function
	#                                                          
	#	a function to update the model parameters, e.g. W1, b1, W2, b2  
	#   Within the function, loss and gradients will be calculated.
	#   This function can only be called by train(), rather than 
	#   independently called
	#                                                                 
	### -------------------------------------------------------- ####
	def _update(self):
		
		# 1. random choose batch_size sub-training points
		num_train = self.X_train.shape[0]
		
		# 2. randomly choose -- batch_mask
		batch_mask = np.random.choice(num_train, self.batch_size)
		
		# 3. select the batch sub training set
		X_batch = self.X_train[batch_mask]
		y_batch = self.y_train[batch_mask]
		
		# 4. calculate loss and grads based on neural network class
		loss, grads = self.model.loss(X_batch, y_batch)
		self.loss_history.append(loss)

		# 5. perform a parameter update based on an optimization approch
		# and relevant optimization arguments, i.e. momentum
		for nn_para_name, nn_para_val in self.model.params.items():
			dw = grads[nn_para_name]
			opt_config = self.optim_nn_params_configs[nn_para_name]
			new_nn_para_val, new_nn_para_opt_config = self.update_rule(nn_para_val, dw, opt_config)
			self.model.params[nn_para_name] = new_nn_para_val
			self.optim_nn_params_configs[nn_para_name] = new_nn_para_opt_config
	
	### -------------------------------------------------------- ####
	#        
	#   check accuracy of current model
	#                                                          
	#	a function to Check accuracy of the model on the provided data
	#                                                                 
	### -------------------------------------------------------- ####	
	def check_error(self, X, y, sub_samples = None, batch_size = 100):
		'''
			Inputs:
			- X: Array of data, of shape (N, d_1, ..., d_k)
			- y: Array of labels, of shape (N,)
			- sub_samples: If not None, subsample the data and only test the model
			on num_samples datapoints.
			- batch_size: Split X and y into batches of this size to avoid using too
			much memory.
      
			Returns:
		    - error: Scalar giving the fraction of instances that were incorrectly
			classified by the model.
		'''
		
		# 1. determine if it needs to sample the subset of data points
		num_data = X.shape[0]
		
		# 1.1 -- if sub_samples is True and number of data points fed is greater
		# than the size of subset
		if sub_samples is not None and num_data > sub_samples:
			mask = np.random.choice(num_data, sub_samples)
			num_data = sub_samples
			
			X = X[mask]
			y = y[mask]
			
		# 2. calculate predictions in batches
		epochs = int(num_data / batch_size)
		
		if num_data % batch_size != 0:
			epochs += 1
		
		# 3. make prediction
		ypred = []
		
		for n in range(epochs):
			start = n * batch_size
			end   = (n+1) * batch_size
			# 3.1 we don't need to evaluate loss, so the very last layer
			# to calculate loss value (such that backprop can be performed)
			# is excluded.
			scores = self.model.loss(X[start:end], None)
			ypred.append(np.argmax(scores, axis = 1))
		
		# 3.2 after 'slice' over all 'windows' (epochs), we concanate all predictions
		ypred_epochs = np.hstack(ypred)
		
		# 3.3 evaluate the error
		error = 1 - np.mean(ypred_epochs == y)
		
		return error
	
	### -------------------------------------------------------- ####
	#        
	#   train function
	#                                                          
	#	run optimization to train the model
	#                                                                 
	### -------------------------------------------------------- ####
	def train(self):
		# 1. retreive number of training points
		num_train = self.X_train.shape[0]
		
		# 2. decide how many iteraions we need to perform at each epoch
		iterations_per_epoch = max(num_train / self.batch_size, 1)
		
		# 3. decide how many iterations in total we need to perform
		num_iterations = int(iterations_per_epoch * self.num_epochs)
		
		# 4. main-loop
		for i in range(num_iterations):
			# 4.1 run sub-train function to update loss, grads			
			self._update()
			
			# 4.2 print out status if necessay
			if self.verbose and i % self.print_every == 0:
				print('(Iteration %d / %d) loss: %f' % (i+1, num_iterations, self.loss_history[-1]))
			
			# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			# 4.3 decay learn_rate at every end of epoch
			# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			end_epoch =  (i + 1) % iterations_per_epoch == 0
			
			if end_epoch:
				self.current_epoch += 1
				
				# 4.3.1 loop-over all optim_config for all parameters in neural network model
				for nn_para_opt_config in self.optim_nn_params_configs:
					self.optim_nn_params_configs[nn_para_opt_config]['learning_rate'] *= self.lr_decay

					
			# 4.4 check error rate at first and end iterations, and the end of 
			# each epoch
			first_iter = (i == 0)
			last_iter  = (i == num_iterations + 1)
			
			if first_iter or last_iter or end_epoch:
				train_error = self.check_error(self.X_train, self.y_train, sub_samples=1000)
				valid_error = self.check_error(self.X_valid, self.y_valid)
				self.train_err_history.append(train_error)
				self.valid_err_history.append(valid_error)
			
				# 4.5 print out status if allowed
				if self.verbose:
					print('(Epoch %d / %d train err: %f; val_err: %f)' \
					%(self.current_epoch, self.num_epochs, train_error, valid_error))

			
			
				# 5. update the optim model -- keep track of the best model
				if valid_error < self.best_valid_err:
					self.best_valid_err = valid_error
					self.best_params = {}
					for nn_para_name, nn_para_val in self.model.params.items():
						self.best_params[nn_para_name] = nn_para_val.copy()
		
		# after loop-over all iterations, update model to best valid accuracy		
		self.model.params = self.best_params			
					
if __name__ == '__main__':
	
	model = twoLayerModularNN(dim_input = 3072,num_hidden=200, num_class = 10, weight_scale = 1e-3, reg=0.5)
	Solver = None	
	Solver = solver(model, data,
              update_rule='sgd',
              optim_config={
                'learning_rate': 1e-3,
              },
              lr_decay=0.95,
              num_epochs=9, batch_size=200,
              print_every=100)
	Solver.train()
	
	# Run this cell to visualize training loss and train / val accuracy
	
	plt.subplot(2, 1, 1)
	plt.title('Training loss')
	plt.plot(Solver.loss_history)
	plt.xlabel('Iteration')
	
	plt.subplot(2, 1, 2)
	plt.title('Error')
	plt.plot(Solver.train_err_history, '-o', label='train')
	plt.plot(Solver.valid_err_history, '-o', label='val')
	#plt.plot([0.5] * len(Solver.valid_err_history), 'k--')
	plt.xlabel('Epoch')
	plt.legend(loc='lower right')
	plt.gcf().set_size_inches(15, 12)
	plt.show()
