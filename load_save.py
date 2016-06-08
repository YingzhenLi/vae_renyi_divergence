import sys, os
import numpy as np
import tensorflow as tf

def path_name(dataset, alpha, num_samples, backward_pass, extra_string = None):
    
    path = 'ckpts/' + dataset + '/'
    if backward_pass == 'max':
        folder_name = 'max_k%d' % num_samples
    else:
        folder_name = 'alpha%.2f_k%d' % (alpha, num_samples)
    if extra_string is not None:
        folder_name += '_' + extra_string
        
    path = path + folder_name + '/'    
    return path
    
def save_checkpoint(sess, path, checkpoint=1, var_list = None):
    if not os.path.exists(path):
        os.makedirs(path)  
    # save model  
    fname = path + 'checkpoint%d.ckpt' % checkpoint
    saver = tf.train.Saver(var_list) 
    save_path = saver.save(sess, fname)
    print("Model saved in %s" % save_path)
    
def load_checkpoint(sess, path, checkpoint=1):
    # load model      
    try:   
        fname = path + 'checkpoint%d.ckpt' % checkpoint
        saver = tf.train.Saver()        
        saver.restore(sess, fname)
        print("Model restored from %s" % fname)
    except:
        print "Failed to load model from %s" % fname
        
