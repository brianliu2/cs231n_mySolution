# -*- coding: utf-8 -*-
"""
These are codes for implementing dropout in forward/backward pass.

@author: xliu
"""

import numpy as np
from gradient_check import *
from class_fullyConnectedNet import *
def dropout_forward(X, dropout_param):
	'''
	Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
			Higher probability means less points are dropped out.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
	'''
	# 1. retrieve the mode of performing dropout
	mode = dropout_param['mode']
	perc = dropout_param.get('perc', 0.5)
	
	# to run gradient check, set seed to generator
	if 'seed' in dropout_param.keys():
		np.random.seed(dropout_param['seed'])
		
	# 2. if the current mode is train
	if mode == 'train':
		mask = (np.random.rand(*X.shape) < perc) / perc
		dropout_out = X * mask
	elif mode == 'test':
		mask = None
		dropout_out = X
	else:
		raise ValueError('dropout has to be train/test.')
	
	# 3. store input information of drop out layer
	dropout_input_info = (dropout_param, mask)
	
	# 4. standardize data type of output and input
	dropout_out = dropout_out.astype(X.dtype, copy = False)
	
	return dropout_out, dropout_input_info



##### dropout in backward pass
def dropout_backward(dout, cache):
	'''
	Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
	'''
	# 0. intialize gradient
	dx = None
	
	# 1. retrieve dropout input information
	dropout_param, mask = cache
	
	# 2. perform dropout in backward pass based on its mode
	mode = dropout_param['mode']
	
	# 3. if the mode is train
	if mode == 'train':
		dx = dout * mask
	elif mode == 'test':
		dx = dout
	
	return dx





###########################################################
#
#
#	testing code
#
#
###########################################################
if __name__ == '__main__':
	
#	# -- test forward pass
#	np.random.seed(231)
#	x = np.random.randn(500, 500) + 10
#	
#	for p in [0.3, 0.6, 0.75]:
#	  out, _ = dropout_forward(x, {'mode': 'train', 'perc': p})
#	  out_test, _ = dropout_forward(x, {'mode': 'test', 'perc': p})
#	
#	  print('Running tests with p = ', p)
#	  print('Mean of input: ', x.mean())
#	  print('Mean of train-time output: ', out.mean())
#	  print('Mean of test-time output: ', out_test.mean())
#	  print('Fraction of train-time output set to zero: ', (out == 0).mean())
#	  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
#	  print()
	
#	# -- test backward pass
#	np.random.seed(231)
#	x = np.random.randn(10, 10) + 10
#	dout = np.random.randn(*x.shape)
#	
#	dropout_param = {'mode': 'train', 'perc': 0.8, 'seed': 123}
#	out, cache = dropout_forward(x, dropout_param)
#	dx = dropout_backward(dout, cache)
#	dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
#	
#	print('dx relative error: ', rel_error(dx, dx_num))

#	# -- test fully connect net with dropout
#	np.random.seed(231)
#	N, D, H1, H2, C = 2, 15, 20, 30, 10
#	X = np.random.randn(N, D)
#	y = np.random.randint(C, size=(N,))
#	
#	for dropout in [0, 0.25, 0.5]:
#	  print('Running check with dropout = ', dropout)
#	  model = fullyConnectedNet([H1, H2], input_dims=D, num_classes=C,
#	                            weight_scale=5e-2, dtype=np.float64,
#	                            dropout=dropout, seed=123)
#	
#	  loss, grads = model.loss(X, y)
#	  print('Initial loss: ', loss)
#	
#	  for name in sorted(grads):
#	    f = lambda _: model.loss(X, y)[0]
#	    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#	    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#	  print()

	# -- test with Cirfa10 dataset
	# Train two identical nets, one with dropout and one without
	np.random.seed(231)
	num_train = 500
	small_data = {
	  'X_train': data['X_train'][:num_train],
	  'y_train': data['y_train'][:num_train],
	  'X_val': data['X_val'],
	  'y_val': data['y_val'],
	}
	
	Solvers = {}
	dropout_choices = [0, 0.75]
	for dropout in dropout_choices:
	  model = fullyConnectedNet(hidden_dims=[500], input_dims=32*32*3,\
			num_classes = 10, dropout=dropout, weight_scale = 1e-2)
	  print(dropout)
	
	  Solver = solver(model, small_data,
	                  num_epochs=25, batch_size=100,
	                  update_rule='adam',
	                  optim_config={
	                    'learning_rate': 5e-4,
	                  },
	                  verbose=True, print_every=100)
	  Solver.train()
	  Solvers[dropout] = Solver
			
	# Plot train and validation accuracies of the two models
	
	train_errs = []
	val_errs = []
	for dropout in dropout_choices:
	  solver = Solvers[dropout]
	  train_accs.append(solver.train_err_history[-1])
	  val_accs.append(solver.valid_err_history[-1])
	
	plt.subplot(3, 1, 1)
	for dropout in dropout_choices:
		plt.plot(Solvers[dropout].train_err_history, '-', label='%.2f dropout' % dropout)
	
	plt.title('Train error')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(ncol=2, loc='lower right')
		  
	plt.subplot(3, 1, 2)
	for dropout in dropout_choices:
		plt.plot(Solvers[dropout].valid_err_history, '-', label='%.2f dropout' % dropout)
	plt.title('Valid error')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(ncol=2, loc='lower right')
	
	plt.subplot(3, 1, 3)
	for dropout in dropout_choices:
		plt.plot(Solvers[dropout].loss_history, '-', label='%.2f dropout' % dropout)
	plt.title('loss')
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.legend(ncol=2, loc='lower right')
	
	plt.gcf().set_size_inches(15, 15)
	plt.show()










































