import sys
import import_data_alphadigs as alphadigs
import import_data_mnist as mnist
import numpy as np
import cPickle
import argparse
from scipy.io import loadmat

def load_data(dataset, path, ratio = 0.9, seed = 0, return_labels = False):
    if dataset == 'freyface':
        data_train, data_test = load_data_freyface(path, ratio, seed)
    if dataset == 'alphadigits':
        data_train, data_test, labels_train, labels_test = \
            load_data_alphadigits(path, ratio, seed)
    if dataset == 'mnist':
        data_train, data_test, labels_train, labels_test = \
            load_data_mnist(path, ratio, seed)
    if dataset == 'silhouettes':
        data_train, data_test = load_data_silhouettes(path, ratio, seed)
    if dataset == 'omniglot':
        data_train, data_test, labels_train, labels_test = \
            load_data_omniglot(path, ratio, seed)
    if return_labels and dataset not in ['freyface', 'silhouettes']:
        return data_train, data_test, labels_train, labels_test
    else:
        return data_train, data_test

def load_data_freyface(path, ratio = 0.9, seed = 0):
    # load and split data
    print "Loading data"
    f = open(path + 'freyface/freyfaces.pkl','rb')
    data = cPickle.load(f)
    data = np.array(data, dtype='f')	# float32
    f.close()
    
    np.random.seed(seed)
    np.random.shuffle(data)
    num_train = int(ratio * data.shape[0])
    data_train = data[:num_train]
    data_test = data[num_train:]
    
    return data_train, data_test
    
def load_data_alphadigits(path, ratio = 0.9, seed = 0):
    # load and split data
    print "Loading data"
    data_train, labels_train, data_test, labels_test = \
        alphadigs.read(path, int(39 * ratio), SEED = seed)
    # transform to float32
    data_train = np.array(data_train.T, dtype='f')	# float32
    data_test = np.array(data_test.T, dtype='f')	# float32
    labels_train = np.array(labels_train.T, dtype='f')	# float32
    labels_test = np.array(labels_test.T, dtype='f')	# float32
    
    return data_train, data_test, labels_train, labels_test

def load_data_omniglot(path, ratio = 0.9, seed = 0):
    # load and split data
    print "Loading data"
    mat = loadmat(path + 'OMNIGLOT/chardata.mat')
    data_train = np.array(mat['data'].T, dtype='f')     # float32
    data_test = np.array(mat['testdata'].T, dtype='f')  # float32
    labels_train = np.array(mat['target'].T, dtype='f') # float32
    labels_test = np.array(mat['testtarget'].T, dtype='f')      # float32

    return data_train, data_test, labels_train, labels_test
    
def load_data_mnist(path, ratio = 0.9, seed = 0, digits = None):
    # load and split data
    print "Loading data"
    path = path + 'MNIST/'
    data_train, labels_train = mnist.read(path, 0, "training", seed, digits)
    data_test, labels_test = mnist.read(path, 0, "testing", seed, digits)
    #data_train = np.array(data >= 0.5 * np.max(data, 0), dtype = int)	# binary
    #data_test = np.array(data >= 0.5 * np.max(data, 0), dtype = int)	# binary
    data_train /= 255.0	# real-value
    data_test /= 255.0	# real-value
    # transform to float32
    data_train = np.array(data_train.T, dtype='f')	# float32
    data_test = np.array(data_test.T, dtype='f')	# float32
    labels_train = np.array(labels_train.T, dtype='f')	# float32
    labels_test = np.array(labels_test.T, dtype='f')	# float32
    return data_train, data_test, labels_train, labels_test

def load_data_silhouettes(path, ratio = 0.9, seed = 0):
    import scipy.io
    imgs_filename = path + 'silhouettes/' \
        + 'caltech101_silhouettes_28_split1.mat'
    with open(imgs_filename, 'rb') as f:
        images = scipy.io.loadmat(imgs_filename)

        images_train = images['train_data'].astype('float32')
        images_test = images['test_data'].astype('float32')
        images_val = images['val_data'].astype('float32')
        #n_validation = images_val.shape[0]
        #images_train = np.vstack((images_train, images_val))
            
    # flip digits?
    images_train = 1.0 - images_train
    images_test = 1.0 - images_test

    return images_train, images_test#, n_validation    
    
