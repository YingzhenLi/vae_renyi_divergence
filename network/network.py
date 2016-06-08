import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from stochastic_layer import construct_Stoc_Layer
from deterministic_layer import Deterministic_Layer, MLP

np.random.seed(0)
tf.set_random_seed(0)

class Network(object):
    """
    A stochastic network containing several stochastic layers.
    """
    def __init__(self, S_layers):
        self.S_layers = S_layers
        self.params = []
        for layer in self.S_layers:
            self.params = self.params + layer.params
        
    def encode(self, input, sampling):
        output = input
        for layer in self.S_layers:
            output, _ = layer.encode(output, sampling)
        return output
        
    def encode_and_log_prob(self, input, eval_output = None):
        # evaluate on eval_output if provided
        if eval_output is None:
            eval_output = [None for layer in self.S_layers]
        output = input
        output_list = []
        l = 0
        for layer in self.S_layers:
            output, logprob = layer.encode_and_log_prob(output, eval_output[l])
            if eval_output[l] is not None:
                output = eval_output[l]
            output_list.append(output)
            if l == 0:
                logprob_total = logprob
            else:
                logprob_total = logprob_total + logprob
            l += 1
        return output_list, logprob
    
    def get_prob_type(self):
        return self.S_layers[-1].get_prob_type()

    def get_name(self):
        return 'stochastic_network'

def construct_network(layer_sizes, prob_type = 'gaussian', data_type='real', \
        activation='softplus', prefix = 'p'):
    """
    Construct a stochastic network.
    """
    S_layers = []
    l = 0
    for sizes in layer_sizes:
        if l == len(layer_sizes) - 1:
            if data_type == 'real': prob_type = 'gaussian'
            if data_type == 'bool': prob_type = 'bernoulli'
        print prefix, l, data_type, prob_type
        S_layers.append(construct_Stoc_Layer(sizes, prob_type, activation, prefix))
        l += 1
    network = Network(S_layers)
    return network  
    
def construct_mlp(layer_sizes, activation = 'softplus'):
    """
    Construct a deterministic network
    """
    D_layers = []
    L = len(layer_sizes) - 1
    for l in xrange(L):
        D_layers.append(Deterministic_Layer(layer_sizes[l], layer_sizes[l+1], activation))
    
    mlp = MLP(D_layers)
    return mlp        
            
