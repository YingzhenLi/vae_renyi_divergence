import numpy as np
import tensorflow as tf
from __init__ import Prior
        
class Gaussian_diag(Prior):

    def __init__(self, size, Mu, Sigma):
        self.size = size
        self.Mu = Mu
        self.Sigma = Sigma
        
    def sample(self, num_samples):
        eps = tf.random_normal([num_samples, self.size])
        output = self.Mu + eps * self.Sigma          
        return output
    
    def update(self, samples):
        # update the parameters by matching empirical moments
        mean, var = tf.nn.moments(samples, axes=[0])
        self.Mu = mean
        self.Sigma = tf.sqrt(var)
        
    def get_name(self):
        return 'prior_gaussian_diag'
        
class Gaussian_full(Prior):

    def __init__(self, Mu, Sigma):
        self.Mu = Mu
        self.Sigma = Sigma
        
    def sample(self, num_samples):
        eps = tf.random_normal([num_samples, self.size])
        output = self.Mu + tf.matmul(eps, self.Sigma)          
        return output
    
    def get_name(self):
        return 'prior_gaussian_full'
        
