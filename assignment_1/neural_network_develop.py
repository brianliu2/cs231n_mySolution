# -*- coding: utf-8 -*-
"""
This is for sub/mini program for developing full neural network

@author: xliu
"""
from neural_network_class import *
import matplotlib.pyplot as plt

# 1. --- define dimensions of neural network -- #
dim_inputs = 5
num_inputs_instances = 4
num_hidden_unit = 10
num_classes = 3


# 2. --- initialize the weights for units in different layers
def init_toy_model():
	np.random.seed(0)
	return neural_network(num_inputs_instances, num_hidden_unit,
							num_classes, std = 1e-1)

# 3. --- generate toy examples data
def generate_toyData(x_dim, x_num):
	np.random.seed(1)
	X = 10 * np.random.randn(x_dim, x_num)
	y = np.array([0, 1, 2, 2, 1])
	return X, y

net = init_toy_model()
X, y = generate_toyData(dim_inputs, num_inputs_instances)

# 4. --- forward pass: compute score

score = net.loss(X, None, reg=0.0)
print(score)


# 5. --- forward pass: compute loss
loss,_ = net.loss(X, y, reg=0.05)
print(loss)

# 6. --- backward pass: back propagation

loss, grads = net.loss(X, y, reg = 0.05)
print(grads)

# 6.1 --- gradient check
def rel_error(x, y):
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


from eval_numerical_gradient import eval_numerical_gradient
for param_name in grads:
	f = lambda w: net.loss(X, y, reg=0.05)[0]
	param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
	print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# 7.1 create a neural network class
nn_clf = neural_network(num_inputs_instances, num_hidden_unit,num_classes, std = 1e-1)
kwargs = {'reg':5e-6, 'learn_rate':1e-3, 'learn_rate_decay':0.95,
	       'numIters': 100, 'batch_size':200}
trainhist = nn_clf.train(X, y, X, y, **kwargs)
trainpred = nn_clf.predict(X)

import neural_network_class
from importlib import reload
reload(neural_network_class)
from neural_network_class import *

plt.plot(trainhist['loss_history'])




















