import sys
sys.path.append('../')
from init_models import *
import numpy as np
import argparse
from data.data_preprocessing import load_data
from load_save import *
from visualization.reconstruction import plot_recon
from visualization.samples import plot_samples

def main(dataset, dimZ, hidden_layers, n_iters, learning_rate = 0.0005, \
        batch_size = 100, seed = 0, alpha = 1.0, num_samples = 1, \
        save = False, backward_pass = 'full', activation = 'softplus', \
        loss = 'vae', checkpoint = 0):
    
    # load data
    ratio = 0.9   
    path = 'data/'
    supervised = False
    data_train, data_test = load_data(dataset, path, ratio, seed, supervised)
    if dataset == 'freyface':
        data_type = 'real'
    else:
        data_type = 'bool'
        
    # initialise the computation
    sess = tf.Session()
    variables_size = [data_train.shape[1], dimZ]
    
    # TODO: other training methods coming soon...
    if loss == 'vae':
        kwargs = {'alpha': alpha, 'backward_pass': backward_pass}
        print 'training model: variational auto-encoder' 
        from models.vae import init_optimizer       

    models = init_model(variables_size, hidden_layers, data_type, activation)
    prior = init_prior_gaussian(variables_size[-1])
    if checkpoint > 0:
        path = path_name(dataset, alpha, num_samples, backward_pass)       
        load_checkpoint(sess, path, checkpoint)
        initialised_var = set(tf.all_variables())
    else:        
        initialised_var = set([])
    
    fit, score = init_optimizer(models, variables_size[0], batch_size, \
                                num_samples, **kwargs)
    
    # now check init
    init_var_list = set(tf.all_variables()) - initialised_var
    if len(init_var_list) > 0:
        # Initializing the tensor flow variables
        init = tf.initialize_variables(var_list = init_var_list)
        #init = tf.initialize_all_variables()
        sess.run(init)
    checkpoint += 1

    num_iter_trained = 0
    print "Training..."
    for n_iter in n_iters:
        fit(sess, data_train, n_iter, learning_rate)
        num_iter_trained += n_iter
        print "Evaluating test data..."
        lowerbound_test, time_test = \
            score(sess, data_test, num_samples = 10)
        print "test data LL (lowerbound) = %.2f, time = %.2fs, iter %d" \
            % (lowerbound_test, time_test, num_iter_trained)
    
    # plot reconstructions
    if dataset == 'freyface':
        shape = (28, 20)
    if 'mnist' in dataset:
        shape = (28, 28)
    print 'ploting reconstructions...'
    recon_input = data_test[:100]
    plot_recon(sess, recon_input, shape, models[0], models[1])
    
    print 'ploting samples from the generative model...'
    plot_samples(sess, shape, prior, models[1])
    
    # save model
    if save:
        path = path_name(dataset, alpha, num_samples, backward_pass)       
        save_checkpoint(sess, path, checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--data', '-D', type=str, default='freyface')
    parser.add_argument('--num_layers', '-l', type=int, choices=[1, 2], default=1)
    parser.add_argument('--num_samples', '-k', type=int, default=1)
    parser.add_argument('--alpha', '-a', type=float, default=1.0)
    parser.add_argument('--dimZ', '-Z', type=int, default=5)
    parser.add_argument('--dimH', '-H', type=int, default=200)
    parser.add_argument('--iter', '-i', type=int, default=100)
    parser.add_argument('--save_model', '-s', action='store_true', default=False)
    parser.add_argument('--seed', '-S', type=int, default=0)
    parser.add_argument('--backward_pass', '-b', type=str, default='full')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--activation', type=str, default='softplus')
    parser.add_argument('--loss', type=str, default='vae')
    parser.add_argument('--checkpoint', type=int, default=0)
    
    args = parser.parse_args()
    if args.dimH > 0:
        hidden_layers = [[args.dimH for i in xrange(args.num_layers)]]
    else:
        hidden_layers = [[]]
    if args.backward_pass not in ['full', 'single', 'max', 'min']:
        args.backward_pass = 'full'
    
    print 'settings:'
    print 'activation function:', args.activation
    print 'dataset:', args.data
    print 'alpha:', args.alpha
    print 'dimZ:', args.dimZ
    print 'hidden layer sizes:', hidden_layers
    print 'num. samples:', args.num_samples
    print 'backward pass method:', args.backward_pass
    print 'learning rate:', args.learning_rate
    print 'batch_size:', args.batch_size
    
    iter_each_round = 10
    num_rounds = args.iter / iter_each_round
    n_iters = list(np.ones(num_rounds, dtype = int) * iter_each_round)
    main(args.data, args.dimZ, hidden_layers, n_iters, args.learning_rate, \
        args.batch_size, args.seed, args.alpha, args.num_samples, \
        args.save_model, args.backward_pass, args.activation,
        args.loss, args.checkpoint)
    
