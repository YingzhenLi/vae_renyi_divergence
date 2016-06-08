import numpy as np
import tensorflow as tf
from deterministic_layer import Deterministic_Layer

def construct_classifier(layer_sizes, activation='softplus'):
    """
    Construct the stochastic layer.
    """
    D_layers = []
    for l in xrange(len(layer_sizes) - 1):
        if l < len(layer_sizes) - 2:
            func = 'relu'
        else:
            func = activation
        D_layers.append(Deterministic_Layer(layer_sizes[l], layer_sizes[l+1], func))
    
    classifier = Classifier(D_layers)   
    return classifier

class Classifier(object):

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
        return 'classifier'
        
