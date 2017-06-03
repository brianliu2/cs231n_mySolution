# -*- coding: utf-8 -*-
"""
This is the class of neural netwrok.
More specifically, this is a fully connected 
two layers neural network. The input of network
is in dimension of N, a hidden layer dimension of H,
and performs classification over C classes.

We train the network with a softmax loss function and 
L2 regularization on the weight matrices. The network
uses a ReLu nonlinearity activation after the first
fully connected layers.

Architecture of this network is:

input --> inputs * weights_1 --> ReLu --> outputs * weights_2 --> softmax

Input:
	input_size -- the number of input instances for netural network
	hidden_size -- the number of units of hidden layer
	output_size -- the number of outputs in output layer
	
Output:
	scores: scpre for each class
"""
import numpy as np

class neural_network(object):
	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	# implementation for calculating loss/gradient
	def loss(self, X, y, reg = 0):
		# 1. initialize
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		
		num_input_instances = X.shape[0]
		num_features = X.shape[1]
		
		loss = None
		
		# 2. computing scores in forwardfeed
		layer1_input  = X.dot(W1) + b1
		layer1_output = np.maximum(0, layer1_input)
		layer2_intput = layer1_output.dot(W2) + b2
		scores = layer2_intput
		
		# output scores if y is None
		if y is None:
			return scores
		
		# 3. compute the loss
		# --- 3.1 compute the exponential score
#		scores -= np.max(scores, axis = 1, keepdims = True)
		exp_score = np.exp(scores)
		
		# --- 3.2 compute the sum of exp_score for getting denominator
		denom_exp_score = np.sum(exp_score, axis = 1, keepdims = True)
		
		# --- 3.3 'normalized' probabilities
		prob_cls = exp_score / denom_exp_score
		
		# --- 3.4 probabilities for 'correct' classes
		prob_correct_cls = prob_cls[np.arange(num_input_instances), y]
		
		# --- 3.5 'cross-entropy': negative log of probabilities of correct classes
		neglog_prob_correct_cls = -np.log(prob_correct_cls)
		
		# --- 3.6 loss value without regularization
		loss = np.sum(neglog_prob_correct_cls) / float(num_input_instances)
		
		# --- 3.7 count the regularization term
		regloss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
		
		# --- 3.8 loss = loss + regularization
		loss += regloss
		## ----------------------------------------------
		#
		#
		#  gradients part:
		#	
		#	four gradients need to be computed.
		#
		#	1 -- W2
		#	2 -- b2
		#	3 -- W1
		# 	4 -- b1
		#
		#
		## ----------------------------------------------
		# --- 4.1 initialize gradients dictionary
		grads = {}
		
		# --- 4.2: dw2 = dz6/dz5 * dz5/dz4 * dz4/dw2
		# ---    : db2 = dz6/dz5 * dz5/dz4 * dz4/db2
		grad_prob = prob_cls
		grad_prob[np.arange(num_input_instances), y] -= 1
		
		# !!!!!!!!!!!!!!!!! divide by number of instances
		grad_prob /= num_input_instances
		
		grads['W2'] = np.dot(layer1_output.T, grad_prob)
		grads['b2'] = np.sum(grad_prob, axis = 0)
		
		# --- 4.3: dw1 = dz6/dz5 * dz5/dz4 * dz4/dz3 * dz3/dz2 * dz2/dw1
		# ---    : db1 = dz6/dz5 * dz5/dz4 * dz4/dz3 * dz3/dz2 * dz2/db1
		dhidden = np.dot(grad_prob, W2.T)
		
		# 4.3.1 derivative of ReLu wrt 'output' from ReLu activation
		# is either 1 or 0
		dhidden[layer1_output <= 0] = 0
		
		# 4.4 dw1
		grads['W1'] = np.dot(X.T, dhidden)
		grads['b1'] = np.sum(dhidden, axis = 0)
		
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# regularization
		grads['W2'] += reg * W2
		grads['W1'] += reg * W1	
		
		return loss, grads
		

### -------------------------------------
#
#	train neural network
#
### -------------------------------------

	def train(self, X, y, X_val, y_val, reg, numIters,\
			  batch_size, learn_rate, learn_rate_decay, verbose=False):
		# 1. retrieve information of dataset
		num_train = X.shape[0]
		num_feature = X.shape[1]
		num_classes = len(np.unique(y))
		
		# 2. each epoch, we perform #n iterations
		iterations_per_epoch = max(1, num_train / batch_size)		
				
		# 3. create containers for storing training history
		loss_hist = []
		train_acc_hist = []
		valid_acc_hist = []
		
		
		# 4. loop-over iterations we want to perform
		for i in range(numIters):
			idxs = np.random.choice(num_train, batch_size)
			X_batch, y_batch = X[idxs], y[idxs]
			loss_batch, grads_batch = self.loss(X_batch, y_batch, reg)
			self.params['W1'] += -learn_rate * grads_batch['W1']
			self.params['b1'] += -learn_rate * grads_batch['b1']
			self.params['W2'] += -learn_rate * grads_batch['W2']
			self.params['b2'] += -learn_rate * grads_batch['b2']
			loss_hist.append(loss_batch)
		
			# 5. print-out the current training status
			if verbose and i%500 == 0:
				print('Iterations: %d / %d; loss: %f' % (i, numIters, loss_batch))
			
			# 6. for every epoch (time we sampled batch_size), we decay the learn rate
			#    and check the accuracy
			if i % iterations_per_epoch == 0:
				y_train_pred = self.predict(X_batch)
				train_acc = np.mean(y_train_pred == y_batch)
				train_acc_hist.append(train_acc)
				
				y_valid_pred = self.predict(X_val)
				valid_acc = np.mean(y_valid_pred == y_val)
				valid_acc_hist.append(valid_acc)
				
				# 6.1 decay the learning rate as epoch rolls
				learn_rate *= learn_rate_decay
		
		# 7. return the training information
		train_hist = {'loss_history': loss_hist, 
		'train_accuracy':train_acc_hist, 'validation_accuracy':valid_acc_hist}
		
		return train_hist
	
	# function to make the prediction using trained weights
	def predict(self, X):
		# first layer: linear multiplication + ReLu
		layer_1_input = X.dot(self.params['W1']) + self.params['b1']
		layer_1_output = np.maximum(0, layer_1_input)
		
		# second layer: linear multiplication + bias
		layer_2_input = layer_1_output.dot(self.params['W2']) + self.params['b2']
#		layer_2_input -= np.max(layer_2_input, axis = 1, keepdims = True)
		# output layer: softmax and find the index of max probabilities
		prob_sum = np.sum(np.exp(layer_2_input), axis = 1, keepdims = True)
		prob_cls = np.exp(layer_2_input) / prob_sum
		ypred = np.argmax(prob_cls, axis = 1)
	
		return ypred
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	












