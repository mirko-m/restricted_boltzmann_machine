import numpy as np
import matplotlib.pyplot as plt
import mnist

def sigm(x):
    return 1.0/(1.0 + np.exp(-x))

class RestrictedBoltzmannMachine:
    '''Implementation of binary Restricted Boltzmann Machine (RBM) using numpy.
    Both the visible and the hidden units are binary. The weights are
    initialized using Gauusian ranom numbers and the biases are initialized to
    zero.

    Parameters
    ----------
    n_v: (Integer) Number of visible units
    n_h: (Integer) Number of hidden units.

    Attributes
    ----------
    n_v: (Integer) Number of visible units
    n_h: (Integer) Number of hidden units.
    weights: (Float) Array with shape (n_h, n_v). The weights connecting
        the visible and hidden units.
    bias_v: (Float) Array of length n_v with the biases for the visible units.
    bias_h: (Float) Array of length n_h with the biases for the hidden units.

    Methods
    -------
    sample_h_from_v
    sample_v_from_h
    gibbs_step
    fit
    calc_squared_reconstruction_error

    '''

    def __init__(self,n_v,n_h):
        self.n_v = n_v
        self.n_h = n_h
        self.weights = np.random.randn(n_h,n_v)/100.0
        self.bias_v = np.zeros(n_v)
        self.bias_h  = np.zeros(n_h)

    def sample_h_from_v(self,v):
        '''Sample the hidden units from the visible units.

        Parameters
        ----------
        v: Binary array of shape (n_samples, n_v) representing n_samples
            different configurations of the visible units

        Returns
        -------
        prob_h: (Float) Array of shape (n_samples, n_h) with the probability
            for the hidden units to be turned on.
        h: Binary Array of shape (n_samples, n_h) representing a sample of the
            hidden units drawn from v.

        Note
        ----
        when a single vector v is passed it is important that the shape is
        (1, n_v).

        '''
        dim = v.shape[0]
        prob_h = sigm(np.dot(v,self.weights.T) + self.bias_h) # dimension dim x n_h
        h = (np.random.rand(dim,self.n_h) < prob_h).astype(float)
        return prob_h, h

    def sample_v_from_h(self,h):
        '''Sample the visible units from the visible units.

        Parameters
        ----------
        h: Binary array of shape (n_samples, n_h) representing n_samples
            different configurations of the hidden units

        Returns
        -------
        prob_v: (Float) Array of shape (n_samples, n_v) with the probability
            for the visible units to be turned on.
        v: Binary Array of shape (n_samples, n_v) representing a sample of the
            visible units drawn from h.

        Note
        ----
        when a single vector h is passed it is important that the shape is
        (1, n_v).

        '''
        dim = h.shape[0]
        prob_v = sigm(np.dot(h,self.weights) + self.bias_v) # dimension dim x n_v
        v = (np.random.rand(dim,self.n_v) < prob_v).astype(float)
        return prob_v, v

    def gibbs_step(self,X):
        '''Perform a single Gibbs step which updates first the hidden units and then
        the visible units. Can be used for reconstruction of a sample.

        Parameters
        ----------
        X: Binary array of shape (n_samples, n_v)

        Returns
        -------
        prob_h: (Float) Array of shape (n_samples, n_h) with the probability
            for the hidden units to be turned on.
        h: Binary Array of shape (n_samples, n_h) representing a sample of the
            hidden units drawn from X.
        prob_v: (Float) Array of shape (n_samples, n_v) with the probability
            for the visible units to be turned on given h.
        v: Binary Array of shape (n_samples, n_v) representing a sample of the
            visible units drawn from h, i.e. the reconstruction of X.

        Note
        ----
        when a single vector h is passed it is important that the shape is
        (1, n_h).

        '''
        prob_h, h = self.sample_h_from_v(X)
        prob_v, v = self.sample_v_from_h(h)
        return prob_h, h, prob_v, v

    def fit(self,X,lr,decay=0.0,k=1,n_iter=10,batch_size=100,\
            persistent=True, verbose=True):
        '''Fit the RBM to the data X using k-contrastive divergence (k-CD) or
        k-persistent contrastive divergence (k-PCD). For k-CD the Markov chain
        is reset after each iteration using the training data. For k-CPD on the
        other hand the Markov chain is initialized once using the training
        data, but not reset.

        Parameters
        ----------
        X: Binary array of shape (n_samples, n_v) containing the training data
        lr: the learning rate
        decay: (default = 0) parameter for weight decay to encourage sparsity.
        k: (default = 1) number of samples that are drawn from the Markov-chain
            in each step.
        n_iter: (default = 10) number of iterations. In each iteration every
            sample contained in X is visited once.
        batch_size: (default = 100) the size of the mini_batches used for the
            fitting procedure. After each mini_batch the gradient is updated.
        persistent: (default = True) Use k-PCD or k-CD.
        verbose: (default=True) print iteraton number and squared
            reconstruction error to stdout.

        Note
        ----
        1. The k-CD algorithm implemented here is taken from Algorithm 1 of A.
        Fischer and C. Igel, "Training restricted Boltzmann machines: An
        Introduction", Pattern Recognition 47, 25 (2014). The k-CPD algorithm
        is a slight variation of this algorithm descibed in the same reference.

        2. The squared reconstruction error is not necessarily a good indicator
        of whether the algorithm is truly optimizing the log-likelihood. This
        needs to be kept in mind.

        '''
        n_samples = X.shape[0]
        n_batch = n_samples/batch_size
        n_rest = n_samples % batch_size

        if persistent:
            # Initialize Markov-chain with first batch of training data
            v_k = np.copy(X[:batch_size,:])
        for step in xrange(0,n_iter):
            # If n_samples is not a multiple of n_batch, the final batch has a
            # smaller size. Instead of running the algorithm on the smaller
            # batch I choose to reshuffle the data so that eventually all of
            # the data will be used for fitting.
            shuffled = np.random.permutation(X)
            for j in xrange(0,n_batch):
                X_batch = shuffled[j*batch_size:(j+1)*batch_size,:]
                if persistent:
                    v_k = self._pcd_step(X_batch,v_k,lr,decay=decay,k=k)
                else:
                    self._cd_step(X_batch,lr,k=k)

            if verbose:
                print 'iteration\t{:d}\tsquared_reconstruction_error\t{:.6f}\t'.\
                    format(step, self.calc_squared_reconstruction_error(X))

    def _pcd_step(self,X,v,lr,decay=0.0,k=1):
        '''Helper function which performs a simgle step of k-PCD.

        Parameters
        ----------
        v: Binary array of shape (n_samples, n_v) representing the current
            state of the visible units in the Markov chain.
        For all other parameters see docstring of fit

        Returns
        -------
        v_k: Binary array of shape (n_samples, n_v) representing the visible
            units after updating the Markov chain k times. This is passed back
            into _pcd_step in the next iteration.

        '''
        n_samples = X.shape[0]
        prob_h_0, h_0 = self.sample_h_from_v(X)
        prob_h_k, h_k, prob_v_k , v_k = self.gibbs_step(v)

        # loop below is entered when k>1
        for t in range(1,k):
            prob_h_k, h_k, prob_v_k , v_k = self.gibbs_step(v)

        prob_h_k = sigm(np.dot(v_k,self.weights.T) + self.bias_h)

        # The update rules below were taken from A. Fisher and C. Igel
        # Pattern Recognition 47, 25, (2014)
        self.weights += lr*(np.dot(prob_h_0.T,X) -\
                            np.dot(prob_h_k.T,v_k))/n_samples\
                            - decay*self.weights
        self.bias_h += lr*np.sum(prob_h_0 - prob_h_k,axis=0)/n_samples
        self.bias_v += lr*np.sum(X - v_k,axis=0)/n_samples

        return v_k

    def _cd_step(self,X,lr,decay=0.0,k=1):
        '''Helper function which performs a simgle step of k-CD.

        Parameters
        ----------
        See docstring of fit

        '''
        n_samples = X.shape[0]
        prob_h_0, h_0, prob_v_k , v_k = self.gibbs_step(X)

        # loop below is entered when k>1
        for t in range(1,k):
            prob_h_k, h_k, prob_v_k , v_k = self.gibbs_step(v_k)

        prob_h = sigm(np.dot(v_k,self.weights.T) + self.bias_h)

        # The update rules below were taken from A. Fisher and C. Igel
        # Pattern Recognition 47, 25, (2014)
        self.weights += lr*(np.dot(prob_h_0.T,X) -\
                            np.dot(prob_h_k.T,v_k))/n_samples\
                            - decay*self.weights
        self.bias_h += lr*np.sum(prob_h_0 - prob_h_k,axis=0)/n_samples
        self.bias_v += lr*np.sum(X - v_k,axis=0)/n_samples


    def calc_squared_reconstruction_error(self,X):
        '''Calculate the squared reconstruction error

        Parameters
        ----------
        X: Binary array of shape (n_samples, n_v) containing the data to be
            reconstructed

        Returns
        -------
        squared reconstruction error (Float)

        '''
        _, _, _, X_rec = self.gibbs_step(X)
        return np.mean(np.square(X-X_rec))

    def save_state_to_file(self,fname):
        '''Save the current state (weights and biases) of the RBM to a file.

        Parameters
        ----------
        fmname: (string) name of the file

        '''
        f = open(fname,'w')
        np.savez(f,weights=self.weights,bias_v=self.bias_v,bias_h=self.bias_h)

    def load_saved_state(self,fname):
        '''Load a saved state (weights and biases) from a file

        Parameters
        ----------
        fmname: (string) name of the file

        '''
        data = np.load(fname)
        self.weights = data['weights']
        self.bias_v = data['bias_v']
        self.bias_h = data['bias_h']
