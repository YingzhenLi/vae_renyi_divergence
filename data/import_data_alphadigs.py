import numpy as np
from scipy.io import loadmat

def read(path, num_per_digit_train = 10, SEED = 0):
 
 	# load data
	mat = loadmat(path + 'binaryalphadigs.mat')
	img = mat['dat']
	lbl = mat['classlabels']

	num_class = 36
	num_per_digit_train = min(num_per_digit_train, 39)
	num_per_digit_test = 39 - num_per_digit_train
	num_data_train = num_per_digit_train * num_class
	num_data_test = num_per_digit_test * num_class
    
	rows = 20; cols = 16
	img_train =  np.zeros([rows*cols, num_data_train])
	lbl_train = np.zeros([num_class, num_data_train])
	img_test =  np.zeros([rows*cols, num_data_test])
	lbl_test = np.zeros([num_class, num_data_test])

	np.random.seed(SEED)
	for j in xrange(num_class):
		ind = np.random.permutation(range(39))
		for k in xrange(num_per_digit_train):
			img_train[:, k * num_class + j] = np.ravel(img[j, ind[k]])
			lbl_train[j, k * num_class + j] = 1
		for k in xrange(num_per_digit_test):
			img_test[:, k * num_class + j] = np.ravel(img[j, ind[num_per_digit_train + k]])
			lbl_test[j, k * num_class + j] = 1

	return img_train, lbl_train, img_test, lbl_test
	
