# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:52:30 2017

@author: xliu
"""

import numpy as np
import pickle
import os

def load_CIFAR10(Root):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        file = os.path.join(Root, 'data_batch_%d' % (b, ))
        # read-in training data and loop over all files
        with open(file, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
            # append to subset, and will concatenate after retriving all data
            xs.append(X)
            ys.append(Y)
    # concatenate after retriving all data
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    
    # read-in the test dataset
    fileTest = os.path.join(Root, 'test_batch')
    with open(fileTest, 'rb') as f:
        datadict = pickle.load(f, encoding = 'latin1')
        Xte = datadict['data']
        Yte = datadict['labels']
        Xte = Xte.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Yte = np.array(Yte)
    #return {'X_train': X_train, 'y_train': y_train,
    #        'X_val': X_val, 'y_val': y_val,
    #        'X_test': X_test, 'y_test': y_test}
    return Xtr, Ytr, Xte, Yte
				
				
def subsampleData(X_train, y_train, X_test, y_test, num_training=49000, num_validation=1000, num_test=1000):
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
				
######## self-run ################
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:49:29 2017

@author: xliu
"""
if __name__ == '__main__':
	
	# load-in data
	# define the root of data files
	cifar10_dir = os.path.join('/Users/xliu/Documents/MRC/Work/Online course/',
	    'CS231N Convolutional Neural Networks for Visual Recognition/',
	'assignment1/assignment1/cs231n/datasets/cifar-10-batches-py')
	
	read_data = True
	
	if read_data == True:
		X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
		
		# As a sanity check, we print out the size of the training and test data.
		print('Training data shape: ', X_train.shape)
		print('Training labels shape: ', y_train.shape)
		print('Test data shape: ', X_test.shape)
		print('Test labels shape: ', y_test.shape)
		
		
		data = subsampleData(X_train, y_train, X_test, y_test)
		for k, v in data.items():
		    print('%s: ' % k, v.shape)
	
	print('data has been read.')
