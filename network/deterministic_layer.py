import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

def init_weights(input_size, output_size, constant=1.0, seed=123): 
    """ Glorot and Bengio, 2010's initialization of network weights"""
    scale = constant*np.sqrt(6.0/(input_size + output_size))
    if output_size > 0:
        return tf.random_uniform((input_size, output_size), 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)
    else:
        return tf.random_uniform([input_size], 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)

class Deterministic_Layer(object):
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size       
        # activation function
        self.name = activation
        if activation == 'softplus':
            self._activation = tf.nn.softplus
        if activation == 'relu':
            self._activation = tf.nn.relu
        if activation == 'sigmoid':
            self._activation = tf.sigmoid
        if activation == 'tanh':
            self._activation = tf.tanh
        if activation == 'linear':
            self._activation = lambda x: x
        if activation == 'softmax':
            self._activation = tf.nn.softmax
        # parameters
        W = tf.Variable(init_weights(input_size, output_size))
        b = tf.Variable(tf.zeros([output_size]))
        #b = tf.Variable(init_weights(output_size, 0))
        self.params = [W, b]

    def encode(self, input):
        return self._activation(tf.matmul(input, self.params[0]) + self.params[1])
    
    def get_name(self):
        return self.name

class MLP(object):
    def __init__(self, D_layers):
        self.D_layers = D_layers
        self.params = []
        for layer in self.D_layers:
            self.params = self.params + layer.params

    def encode(self, input):
        output = input
        for layer in self.D_layers:
            output = layer.encode(output)

        return output

    def get_name(self):
        return 'MLP'

