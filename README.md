This repository contains code for an implementation of a binary Restricted
Boltzmann Machines (RBM) using both numpy and tensorflow. The models are trained
using k-contrastive-divergence or k-persistent-contrastive-divergence. A
discussion of these algorithms can be found in Ref \[1\].

Some jupyter notebooks which train the models on the MNIST data set
(http://yann.lecun.com/exdb/mnist/) are also provided. They show that the squared reconstruction error is not necessarily a good measuere of how well the RBM learns the training data. Furthermore I find that persistent-contrastive-divergence yields better results than standard contrastive-divergence.

__Useful References__

\[1\] A. Fischer and C. Igel, _Training restricted Boltzmann machines\: An Introduction_, [Pattern Recognition **47**, 25 (2014).](https://www.sciencedirect.com/science/article/pii/S0031320313002495)

\[2\] G. Hinton, [_A Practical Guide to Training Restricted Boltzmann Machines (Version 1)._](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)

