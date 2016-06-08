import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp

np.random.seed(0)
tf.set_random_seed(0)

def reconstruction_loss(input, encoder, decoder, num_samples):
    """
    Compute log p(x|z) and log q(z|x)
    """
    # compute log_q
    x_rep = tf.tile(input, [num_samples, 1])
    z_list, logq = encoder.encode_and_log_prob(x_rep)
    # compute log_p    
    samples = list(reversed(z_list[:-1]))
    samples.append(x_rep)
    _, logpxz = decoder.encode_and_log_prob(z_list[-1], eval_output = samples)
    
    return logpxz, logq, z_list

def reconstruction_mse_loss(input, encoder, decoder, sampling = False):
    z = encoder.encode(input, sampling)
    input_recon = decoder.encode(z, sampling)
    loss = tf.square(input - input_recon)
    return tf.reduce_mean(tf.reduce_sum(loss, 1)), z
    
def reconstruction_cross_entropy(input, encoder, decoder, sampling = False):
    z = encoder.encode(input, sampling)
    input_recon = decoder.encode(z, sampling)
    loss = -input * tf.log(tf.clip_by_value(input_recon, 1e-9, 1.0)) \
           - (1.0 - input) * tf.log(tf.clip_by_value(1 - input_recon, 1e-9, 1.0))
    return tf.reduce_mean(tf.reduce_sum(loss, 1)), z

def log_prior(z, prob_type = 'gaussian'):
    if prob_type == 'gaussian':
        return log_prior_gaussian(z)
    if prob_type == 'bernoulli':
        return log_prior_bernoulli(z)
    if prob_type == 'bernoulli_sym':
        return log_prior_bernoulli_sym(z)
    if prob_type == 'softmax':
        return log_prior_softmax(z, int(z.get_shape()[0]))
    
def log_prior_gaussian(z, Mu = 0.0, Sigma = 1.0):
    logprob = -(0.5 * np.log(2 * np.pi) + tf.log(Sigma)) \
                  - 0.5 * ((z - Mu) / Sigma) ** 2
    return tf.reduce_sum(logprob, 1)
                  
def log_prior_bernoulli(z, Mu = 0.5):
    logprob = z * tf.log(tf.clip_by_value(Mu, 1e-9, 1.0)) \
                  + (1 - z) * tf.log(tf.clip_by_value(1 - Mu, 1e-9, 1.0))
    return tf.reduce_sum(logprob, 1)
    
def log_prior_bernoulli_sym(z, Mu = 0.5):
    a = (z + 1.0) / 2.0
    logprob = a * tf.log(tf.clip_by_value(Mu, 1e-9, 1.0)) \
                  + (1 - a) * tf.log(tf.clip_by_value(1 - Mu, 1e-9, 1.0))
    return tf.reduce_sum(logprob, 1)

def log_prior_softmax(z, N = 2.0):
    # TODO: implement other logits
    logprob = z * tf.log(1.0 / float(N))
    return tf.reduce_sum(logprob, 1)

def classification_cross_entropy(y_pred, y_data):    
    loss = -y_data * tf.log(tf.clip_by_value(y_pred, 1e-9, 1.0)) \
           - (1 - y_data) * tf.log(tf.clip_by_value(1 - y_pred, 1e-9, 1.0))
    return tf.reduce_mean(loss)

def classification_cross_entropy_softmax(y_pred, y_data):    
    loss = -y_data * tf.log(tf.clip_by_value(y_pred, 1e-9, 1.0))
    return tf.reduce_mean(tf.reduce_sum(loss, 1))

def classification_error_one_hot(y_pred, y_data):
    # assume y_data is a one-hot vector
    # and y_pred contains probabilities
    correct_prediction = tf.equal(tf.argmax(y_data,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return 1.0 - accuracy

def adversarial_loss(y_p, y_q):
    loss = -tf.log(tf.clip_by_value(y_p, 1e-9, 1.0)) \
           - tf.log(tf.clip_by_value(1 - y_q, 1e-9, 1.0))
    return tf.reduce_mean(tf.reduce_sum(loss, 1))
    
