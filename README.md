# Renyi divergence variational inference applied to variational auto-encoders

I provide a tensorflow implementation of VAE training with Renyi divergence. Math details can be found here:

Yingzhen Li and Richard E. Turner. Renyi divergence variational inference. 
(http://arxiv.org/abs/1602.02311)

I only included some small dataset for testing. To add in more datasets, download them somewhere else and then add them to the data/ directory.

For example, you can download:

MNIST dataset 
(http://yann.lecun.com/exdb/mnist/)
and include all the data files in directory data/MNIST/

OMNIGLOT
(https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT)
and include all the data files in directory data/OMNIGLOT/

Frey Face
(https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl)
and include all the data files in directory data/freyface/

To have a quick test (IWAE), run 
python exp.py --data [dataset name] --alpha [alpha value] -k [num of samples] --dimZ [dimension of the latents]

alpha = 1.0 returns the vanila VAE, alpha = 0.0 gives IWAE. If you want to see the max trick, add in one more option --backward_pass max.
