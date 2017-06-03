# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:23:54 2017

@author: xliu
"""
import numpy as np


def eval_numerical_gradient(f, x, verbose = True, h = 0.00001):
	fx = f(x)
	
	grad = np.zeros_like(x)
	
	# iterate over all indexes in x
	it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
	
	while not it.finished:
		
		# evaluate function at x+h
		ix = it.multi_index
		oldval = x[ix]
		
		x[ix] = oldval + h
		fxph  = f(x)
		
		x[ix] = oldval - h
		fxmh  = f(x)
		
		# reset the value of underlying dimension
		x[ix] = oldval
		
		# compute the partial derivative with centered formula
		grad[ix] = (fxph - fxmh) / (2 * h)
		
		if verbose:
			print(ix, grad[ix])
		
		it.iternext()
		
	return grad