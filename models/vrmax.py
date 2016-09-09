import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from network.network import construct_network
from loss_functions import reconstruction_loss
from loss_functions import log_prior
from vae import variational_lowerbound

def vrmax(x, encoder, decoder, num_samples, batch_size):
    """
    Compute the VR-max trick
    """
    logpxz = 0.0
    logqzx = 0.0
    L = len(encoder.S_layers)
    x_rep = tf.tile(x, [num_samples, 1]) 
    input = x_rep

    # do encoding
    samples = []
    for l in xrange(L):
        output, logq = encoder.S_layers[l].encode_and_log_prob(input)
        logqzx = logqzx + logq
        samples.append(output)
        input = output

    # do decoding
    samples = list(reversed(samples))
    samples.append(x_rep)
    for l in xrange(L):
        _, logp = decoder.S_layers[l].encode_and_log_prob(samples[l], eval_output = samples[l+1])
        logpxz = logpxz + logp

    logpz = log_prior(output, encoder.S_layers[l].get_prob_type())
    logF = tf.reshape(logpxz + logpz - logqzx, [num_samples, batch_size])
    
    # now compute the gradients
    # first test automatic gradient computation
    indices = tf.argmax(logF, 0) * batch_size + tf.constant(np.arange(batch_size))
    samples_max = []
    for l in xrange(len(samples)): 
        samples_max.append(tf.gather(samples[l], indices))
        
    # NOT VERY EFFICIENT! RE-COMPUTE THE BOUND
    logpxz_max = 0.0
    logqzx_max = 0.0
    logpz_max = 0.0
    for l in xrange(L):
        _, logp = decoder.S_layers[l].encode_and_log_prob(samples_max[l], eval_output = samples_max[l+1])
        logpxz_max = logpxz_max + logp
        _, logq = encoder.S_layers[L-1-l].encode_and_log_prob(samples_max[l+1], eval_output = samples_max[l])
        logqzx_max = logqzx_max + logq
    logpz_max = log_prior(samples_max[0], encoder.S_layers[L-1].get_prob_type())    
    lowerbound = tf.reduce_mean(logpxz_max + logpz_max - logqzx_max)
    
    return lowerbound
    
def make_functions_vae(models, input_size, num_samples, batch_size, \
        alpha = 1.0, backward_pass = 'full'): 
    encoder, decoder = models  
 
    input = tf.placeholder(tf.float32, [batch_size, input_size])
    lowerbound = vrmax(input, encoder, decoder, num_samples, batch_size)
                                        
    learning_rate_ph = tf.placeholder(tf.float32, shape = [])
    optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate_ph, \
                                   beta1=0.9, beta2=0.999, epsilon=10e-8 \
                                   ).minimize(-lowerbound)
    
    def updateParams(sess, X, learning_rate = 0.0005):
        opt, cost = sess.run((optimizer, lowerbound),
                           feed_dict={input: X,
                                      learning_rate_ph:learning_rate})
        return cost

    return updateParams, lowerbound                               
                                        
def init_optimizer(models, input_size, batch_size = 100, num_samples = 1, **kwargs):
    
    encoder = models[0]; decoder = models[1]
    # vae
    if 'alpha' not in kwargs:
        alpha = 1.0
    else:
        alpha = kwargs['alpha']
    if 'backward_pass' not in kwargs:
        backward_pass = 'full'
    else:
        backward_pass = kwargs['backward_pass']
    updateParams, lowerbound = \
        make_functions_vae(models, input_size, \
                           num_samples, batch_size, \
                           alpha, backward_pass)

    def fit(sess, X, n_iter = 100, learning_rate = 0.0005, verbose = True):
        # first make batches of source data
        [N, dimX] = X.shape        
        N_batch = N / batch_size
        if np.mod(N, batch_size) != 0:
            N_batch += 1      
        print "training the model for %d iterations with lr=%f" % \
            (n_iter, learning_rate)

        begin = time.time()
        for iteration in xrange(1, n_iter + 1):
            iteration_lowerbound = 0
            ind_s = np.random.permutation(range(N))

            for j in xrange(0, N_batch):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                batch = X[ind]
                lowerbound = updateParams(sess, batch, learning_rate)
                iteration_lowerbound += lowerbound * batch_size

            if verbose:
                end = time.time()
                print("Iteration %d, lowerbound = %.2f, time = %.2fs"
                      % (iteration, iteration_lowerbound / N, end - begin)) 
                begin = end
                
        
    def eval_test_ll(sess, X, num_samples):
        lowerbound = sess.run(variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 1.0, 0.0))
        
        return lowerbound

    def score(sess, X, num_samples = 100):
        """
        Computer lower bound on data, following the IWAE paper.
        """
        
        begin = time.time()
        print 'num. samples for eval:', num_samples
        
        # compute log_q
        lowerbound_total = 0
        num_data_test = X.shape[0]
        if num_data_test % batch_size == 0:
            num_batch = num_data_test / batch_size
        else:
            num_batch = num_data_test / batch_size + 1
        
        for i in xrange(num_batch):
            indl = i*batch_size
            indr = min((i+1)*batch_size, num_data_test)
            minibatch = X[indl:indr]
            lowerbound = eval_test_ll(sess, minibatch, num_samples)
            lowerbound_total += lowerbound * (indr - indl)
        
        end = time.time()
        time_test = end - begin
        lowerbound_total = lowerbound_total / float(num_data_test)

        return lowerbound_total, time_test
     
    return fit, score                              
