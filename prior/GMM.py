import numpy as np
import tensorflow as tf
from __init__ import Prior
        
class GMM_diag(Prior):

    def __init__(self, size, Mu_list, Sigma_list, weights = None):
        self.size = size
        self.Mu_list = Mu_list
        self.Sigma_list = Sigma_list
        self.num_mixture = Mu_list.shape[0]
        if weights is None:
            weights = np.ones(self.num_mixture)
        self.weights = weights / np.sum(weights)
        
    def sample(self, num_samples):
        # first select the mixture
        i = np.random.choice(self.num_mixture, num_samples, p = self.weights)
        eps = tf.random_normal([num_samples, self.size])
        output = self.Mu_list[i] + eps * self.Sigma_list[i]          
        return output
        
    def get_name(self):
        return 'prior_GMM_diag'
        
