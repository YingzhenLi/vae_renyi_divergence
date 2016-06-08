import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from network.network import construct_network

np.random.seed(0)
tf.set_random_seed(0)

def construct_autoencoder(variables_size, hidden_layers, \
        data_type='real', activation='softplus'):
    """
    Construct an auto-encoder, return both encoder and decoder
    """   
    layer_sizes = []
    l = 0
    
    # first construct the encoder
    for d_layers in hidden_layers:
        sizes = [variables_size[l]]
        sizes.extend(d_layers)
        sizes.append(variables_size[l+1])
        layer_sizes.append(sizes)
        l += 1
    encoder = construct_network(layer_sizes, 'gaussian', 'real', activation, 'q')
    print 'q network architecture:', layer_sizes
    print 'prob. type of q net:', encoder.get_name()
    
    # then construct the decoder
    layer_sizes = [list(reversed(sizes)) for sizes in layer_sizes]
    layer_sizes = list(reversed(layer_sizes))
    decoder = construct_network(layer_sizes, 'gaussian', data_type, activation, 'p')
    print 'p network architecture:', layer_sizes
    print 'prob. type of p net:', decoder.get_name()
    
    return encoder, decoder                                      
                                        
