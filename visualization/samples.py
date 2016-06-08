import numpy as np
import tensorflow as tf
from __init__ import plot_images   

def plot_samples(sess, shape, prior, decoder):
    """
    Plot the reconstruction of data.
    """
    z = prior.sample(100)
    x = decoder.encode(z, sampling = False)
    
    samples = sess.run(x)
    plot_images(samples, shape, '', 'samples')
