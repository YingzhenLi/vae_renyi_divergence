import numpy as np
import tensorflow as tf
import time
from models.autoencoder import construct_autoencoder
from network.classifier import construct_classifier
from prior.gaussian import Gaussian_diag
from prior.GMM import GMM_diag

def init_model(variables_size, hidden_layers, data_type, activation = 'softplus'):

    # first initialise models
    encoder, decoder = construct_autoencoder(variables_size, hidden_layers, \
                                             data_type, activation)
                                             
    return [encoder, decoder]

def init_prior_gaussian(output_size, mu = 0.0, sigma = 1.0):
    prior = Gaussian_diag(output_size, mu, sigma)
    return prior
    
def init_prior_GMM(output_size, mu_list = None, sigma_list = None, weights = None):
    if weights is None:
        num_mixture = 4
        weights = np.ones(num_mixture)
    num_mixture = weights.shape[0]
    if mu_list is None:
        mu_list = np.random.randn(num_mixture, output_size) * 1.0
    if sigma_list is None:
        sigma_list = np.ones([num_mixture, output_size])
    prior = GMM_diag(output_size, mu_list, sigma_list, weights)
    return prior

def init_classifier(layer_sizes):
    classifier = construct_classifier(layer_sizes, 'sigmoid')    
    return classifier




    
    
