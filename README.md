# Renyi divergence variational inference applied to variational auto-encoders

**Update 2: 14 Sept 2016**

There are two ways to implement IWAE/VAE with other alpha settings (except 
alpha = 1.0 which gives you the vanila VAE). One is to actually compute the
energy as a scalar and let tensorflow work out the rest for you. The previous 
naive implementation (see [vae.py](models/vae.py)) did this. The other follows 
[section 4.2 of the paper](http://arxiv.org/pdf/1602.02311v2.pdf#5), in which
you compute the gradients on **a list of unormalized log importance weights**,
and form the final gradient by computing the weighted average of them.
My internal use numpy code used this strategy.

So as another quick update I also provide the second strategy implementation
in tensorflow. Please see [iwae.py](models/iwae.py) for details -- just a few
lines of changes. This is of almost the same flavor as 
[the theano version](https://github.com/yburda/iwae/blob/master/iwae.py#L142), 
except that they treated VAE/IWAE as different cases. In contrast this tensorflow
code **handles both cases in the same way** as justified by the paper.

If you want to compare both solutions, use --loss vae for the first and 
--loss iwae for the second. You can specify --alpha for both cases and 
--loss iwae also supports --alpha 1.0 (VAE). 
**Some remarks**: First the runtime for both are roughly the same (I have only 
tested them on my laptop). Second the produced results might differ for 
a few nats. This is due to the numerical issues for logsumexp.

Unfortunately, implementing VR-max following similar style of iwae.py still 
does not give you runtime advantage. So I still keep the dirty tryout 
[vrmax.py](models/vrmax.py). Will come back to this -- again stay tuned!

======================================

**Update 1: 09 Sept 2016**

Recently I found that the previous naive implementation in vae.py does not 
give you time savings with the max trick, when compared to my internel use 
numpy version. This is probably because tensorflow/theano does not 
automatically recognize not to compute the gradients of the samples I dropped. 

So as a temporary update I provide a dirty solution (see vrmax.py) that 
collects the max-weight samples and repeats the VAE procedure for them. 
Yes I know it's far from optimized, but at least it already gives you 
~2x speed-up on CPUs (and maybe 1.5x~1.7x on GPUs depending on your settings).

Will come back to this issue -- stay tuned! 

======================================

I provide a tensorflow implementation of VAE training with Renyi divergence. 
Math details can be found here:

Yingzhen Li and Richard E. Turner. Renyi divergence variational inference. 
(http://arxiv.org/abs/1602.02311)

I only included some small dataset for testing. To add in more datasets, 
download them somewhere else and then add them to the data/ directory.

For example, you can download:

MNIST dataset (http://yann.lecun.com/exdb/mnist/)

and include all the data files in directory data/MNIST/

OMNIGLOT (https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT)

and include all the data files in directory data/OMNIGLOT/

Frey Face (https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl)

and include all the data files in directory data/freyface/

To have a quick test, run 

python exp.py --data [dataset name] --alpha [alpha value] -k [num of samples] 
--dimZ [dimension of the latents]

See exp.py file for more options. In particular, alpha = 1.0 returns 
the vanila VAE, alpha = 0.0 gives IWAE. 

If you want to see the max trick, add in one more option --backward_pass max.
