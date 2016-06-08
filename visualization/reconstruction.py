import numpy as np
import tensorflow as tf
from __init__ import plot_images   

def plot_recon(sess, x, shape, encoder, decoder):
    """
    Plot the reconstruction of data.
    """
    input = tf.placeholder(tf.float32, shape=x.shape)
    z = encoder.encode(input, sampling = False)
    x_recon = decoder.encode(z, sampling = False)
    
    input_recon = sess.run(x_recon, feed_dict = {input: x})
    plot_images(input_recon, shape, '', 'recon_sample')
