import numpy as np
import tensorflow as tf
from __init__ import Prior

class Swiss_Roll(Prior):

    def __init__(self, size, center, radius):
        self.size = size
        self.center = center
        self.radius = radius
        
    def sample(self, num_samples):
        unit = tf.sqrt(tf.random_uniform([num_samples, self.size]))
        r = unit * self.radius
        theta = np.pi * 4.0 * unit
                 
        return output
        
    def get_name(self):
        return 'prior_gaussian_diag'

def sample_z_from_swiss_roll_distribution(batchsize, z_dim, label_indices, n_labels, gpu=False):
	def sample(label, n_labels):
		uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
		r = math.sqrt(uni) * 3.0
		rad = np.pi * 4.0 * math.sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, z_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(z_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
	
	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z
