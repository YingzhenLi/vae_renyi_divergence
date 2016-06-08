import numpy as np
import tensorflow as tf
from deterministic_layer import Deterministic_Layer

def construct_Stoc_Layer(layer_sizes, prob_type = 'gaussian', activation='softplus', prefix = 'q'):
    """
    Construct the stochastic layer.
    """
    D_layers = []
    if len(layer_sizes) > 2:
        for l in xrange(len(layer_sizes) - 2):
            D_layers.append(Deterministic_Layer(layer_sizes[l], layer_sizes[l+1], activation))
    
    if prob_type == 'gaussian':
        if prefix == 'p':
            activation = 'sigmoid'
        if prefix == 'q':
            activation = 'linear'
        Mu_layer = Deterministic_Layer(layer_sizes[-2], layer_sizes[-1], activation)
        Log_Sigma_layer = Deterministic_Layer(layer_sizes[-2], layer_sizes[-1], 'linear') 
        S_layer = Gaussian_Stoc_Layer(D_layers, Mu_layer, Log_Sigma_layer)
        
    if prob_type == 'bernoulli':
        Mu_layer = Deterministic_Layer(layer_sizes[-2], layer_sizes[-1], 'sigmoid')
        S_layer = Bernoulli_Stoc_Layer(D_layers, Mu_layer)
    
    if prob_type == 'bernoulli_sym':
        Mu_layer = Deterministic_Layer(layer_sizes[-2], layer_sizes[-1], 'sigmoid')
        S_layer = Bernoulli_sym_Stoc_Layer(D_layers, Mu_layer)
    
    if prob_type == 'softmax':
        Mu_layer = Deterministic_Layer(layer_sizes[-2], layer_sizes[-1], 'softmax')
        S_layer = Softmax_Stoc_Layer(D_layers, Mu_layer)
          
    return S_layer

class Stoc_Layer(object):

    def encode(self, input, sampling):
        raise NotImplementedError()
    
    def log_prob(self, output, params):
        raise NotImplementedError()
        
    def encode_and_log_prob(self, input, eval_output = None):
        # evaluate on eval_output if provided
        if eval_output is None:
            output, params = self.encode(input, sampling = True)
        else:
            _, params = self.encode(input, sampling = False)
            output = eval_output
        logprob = self.log_prob(output, params)
        return output, logprob
        
    def get_name(self):
        return 'stochastic_layer'
        
class Gaussian_Stoc_Layer(Stoc_Layer):

    def __init__(self, D_layers, Mu_layer, Log_Sigma_layer):
        self.D_layers = D_layers
        self.params = []
        for layer in self.D_layers:
            self.params = self.params + layer.params
        self.Mu_layer = Mu_layer
        self.Log_Sigma_layer = Log_Sigma_layer
        self.params = self.params + self.Mu_layer.params
        self.params = self.params + self.Log_Sigma_layer.params
        # output size
        self.output_size = self.Mu_layer.output_size
        
    def encode(self, input, sampling):
        output = input
        for layer in self.D_layers:
            output = layer.encode(output)
        # now compute mu and sigma
        Mu = self.Mu_layer.encode(output)
        Log_Sigma = 0.5 * self.Log_Sigma_layer.encode(output)
        if sampling:
            eps = tf.random_normal(Mu.get_shape())
            output = Mu + tf.exp(Log_Sigma) * eps
        else:
            output = Mu            
        return output, [Mu, Log_Sigma]
        
    def log_prob(self, output, params):
        (Mu, Log_Sigma) = params
        logprob = -(0.5 * np.log(2 * np.pi) + Log_Sigma) \
                  - 0.5 * ((output - Mu) / tf.exp(Log_Sigma)) ** 2
        return tf.reduce_sum(logprob, 1)      

    def get_prob_type(self):
        return 'gaussian'
            
class Bernoulli_Stoc_Layer(Stoc_Layer):

    def __init__(self, D_layers, Mu_layer):
        self.D_layers = D_layers
        self.params = []
        for layer in self.D_layers:
            self.params = self.params + layer.params
        self.Mu_layer = Mu_layer
        self.params = self.params + self.Mu_layer.params
        # output size
        self.output_size = self.Mu_layer.output_size
        
    def encode(self, input, sampling):
        output = input
        for layer in self.D_layers:
            output = layer.encode(output)
        # now compute mu and sigma
        Mu = self.Mu_layer.encode(output)
        if sampling:
            shape = Mu.get_shape()
            eps = tf.random_uniform(shape)
            output = tf.select(eps - Mu <= 0, tf.ones(shape), tf.zeros(shape))
        else:
            output = Mu            
        return output, Mu
        
    def log_prob(self, output, params):
        Mu = params
        logprob = output * tf.log(tf.clip_by_value(Mu, 1e-9, 1.0)) \
                + (1 - output) * tf.log(tf.clip_by_value(1.0 - Mu, 1e-9, 1.0))
        return tf.reduce_sum(logprob, 1)      
    
    def get_prob_type(self):
        return 'bernoulli'

class Bernoulli_sym_Stoc_Layer(Stoc_Layer):

    def __init__(self, D_layers, Mu_layer):
        self.D_layers = D_layers
        self.params = []
        for layer in self.D_layers:
            self.params = self.params + layer.params
        self.Mu_layer = Mu_layer
        self.params = self.params + self.Mu_layer.params
        # output size
        self.output_size = self.Mu_layer.output_size
        
    def encode(self, input, sampling):
        output = input
        for layer in self.D_layers:
            output = layer.encode(output)
        # now compute mu
        Mu = self.Mu_layer.encode(output)
        if sampling:
            shape = Mu.get_shape()
            eps = tf.random_uniform(shape)
            output = tf.select(eps - Mu <= 0, tf.ones(shape), tf.zeros(shape))
        else:
            output = Mu
        output = output * 2.0 - 1.0           
        return output, Mu
        
    def log_prob(self, output, params):
        Mu = params
        z = (output + 1.0) / 2.0
        logprob = z * tf.log(tf.clip_by_value(Mu, 1e-9, 1.0)) \
                + (1 - z) * tf.log(tf.clip_by_value(1.0 - Mu, 1e-9, 1.0))
        return tf.reduce_sum(logprob, 1)      
    
    def get_prob_type(self):
        return 'bernoulli_sym'

class Softmax_Stoc_Layer(Stoc_Layer):

    def __init__(self, D_layers, Logit_layer):
        self.D_layers = D_layers
        self.params = []
        for layer in self.D_layers:
            self.params = self.params + layer.params
        self.Logit_layer = Logit_layer
        self.params = self.params + self.Logit_layer.params
        # output size
        self.output_size = self.Logit_layer.output_size
        
    def encode(self, input, sampling = False):
        output = input
        for layer in self.D_layers:
            output = layer.encode(output)
        # now compute mu and sigma
        Logit = self.Logit_layer.encode(output)
        Logit = tf.log(tf.nn.softmax(Logit))
        output = tf.exp(Logit)	# probability vector
        #if sampling:
        #    shape = output.get_shape()
        #    eps = tf.random_uniform(shape)
        #    diff = output - eps
        #    max_out = tf.reduce_max(diff, 1, keep_dims = True)
        #    output = tf.sign(diff - max_out) + 1.0
        return output, Logit
        
    def log_prob(self, output, params):
        Logit = params
        logprob = output * Logit
        return tf.reduce_sum(logprob, 1)      
    
    def get_prob_type(self):
        return 'softmax'
            
