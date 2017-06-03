# -*- coding: utf-8 -*-
"""
Codes are implementing the numerical evaluation of gradient of 
loss wrt to weights.

Input:
	f -- function to evaluate the loss value
	W -- D * C; weights matrix
	X --

@author: xliu
"""
import random

def gradient_check(f, x, analytic_grad, num_check = 10, h=1e-5):
	'''
	sample a few random features and only return numerical solutions
	in these dimension
	'''
	for i in range(num_check):
		ix = tuple([random.randrange(m) for m in x.shape])
		
		# current feature
		oldEvl = x[ix]
		
		# generate new feature by moving a delta/h ahead
		x[ix] = oldEvl + h
		
		# evaluate new loss based on the new plus feature
		#fxph = f(newEvl_plus)
		fxph = f(x)
		
		# generate new feature by moving a delta/h back
		x[ix] = oldEvl - h
		
		# evaluate new loss value based on the new minus feature
		#fxmh = f(newEvl_minus)
		fxmh = f(x)
		
		# reset x[ix]
		x[ix] = oldEvl
		
		# numerical solution of gradient
		grad_numerical = (fxph - fxmh) / (2 * h)
		
		grad_analytic  = analytic_grad[ix]
		
		# relative difference between two solutions are calculated
		rel_err = abs(grad_numerical - grad_analytic) / abs(grad_numerical + grad_analytic)
		
		print('numerical: %f analytic: %f, relative error: %e' 
		% (grad_numerical, grad_analytic, rel_err))