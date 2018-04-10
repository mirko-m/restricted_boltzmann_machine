import numpy as np
import tensorflow as tf
import mnist # for debugging
import matplotlib.pyplot as plt # for debugging

class RestrictedBoltzmannMachine:
    '''Implementation of binary Restricted Boltzmann Machine (RBM) using
    tensorflow. Both the visible and the hidden units are binary. The weights
    are initialized using Gaussian ranom numbers and the biases are initialized
    to zero. Note that calling tf.global_variables_initializer will
    reinitialize the weighst and biases and thereby loose the state of a
    trained RBM.

    Parameters
    ----------
    n_v: (Integer) Number of visible units
    n_h: (Integer) Number of hidden units.

    Attributes
    ----------
    n_v: (Integer) Number of visible units
    n_h: (Integer) Number of hidden units.
    weights: (tf.float32) tf.Variable with shape (n_h, n_v). The weights
        connecting the visible and hidden units.
    bias_v: (tf.float32) Array of length n_v with the biases for the visible units.
    bias_h: (tf.float32) Array of length n_h with the biases for the hidden
        units.

    Methods
    -------
    sample_h_from_v
    sample_v_from_h
    gibbs_step
    fit
    calc_squared_reconstruction_error

    '''

    def __init__(self,n_v, n_h):
        self.n_v = n_v
        self.n_h = n_h
        self.weights = tf.Variable(tf.random_normal((n_h,n_v),stddev=0.01,dtype=tf.float32),\
                                   name='weights')
        self.bias_v = tf.Variable(np.zeros(n_v),dtype=tf.float32,name='bias_v')
        self.bias_h = tf.Variable(np.zeros(n_h),dtype=tf.float32,name='bias_h')

    def sample_h_from_v(self,v):
        '''Sample the hidden units from the visible units.

        Parameters
        ----------
        v: (tf.float32) Binary tensor of shape (n_samples, n_v) representing n_samples
            different configurations of the visible units. 

        Returns
        -------
        prob_h: (tf.float32) tensor of shape (n_samples, n_h) with the probability
            for the hidden units to be turned on.
        h: (tf.float32) Binary tensor of shape (n_samples, n_h) representing a sample of the
            hidden units drawn from v.

        Note
        ----
        when a single vector v is passed it is important that the shape is
        (1, n_v).

        '''
        # Note shape returns a tensorflow Dimension object. We need the value
        # of that object
        batch_size = v.shape[0].value
        with tf.name_scope('sample_h_from_v'):
            prob_h = tf.nn.sigmoid(tf.matmul(v,self.weights,transpose_b=True) + self.bias_h)
            h = tf.cast(tf.random_uniform((batch_size,self.n_h)) < prob_h,tf.float32)
        return prob_h, h

    def sample_v_from_h(self,h):
        '''Sample the visible units from the hidden units.

        Parameters
        ----------
        h: (tf.float32) Binary tensor of shape (n_samples, n_v) representing n_samples
            different configurations of the hidden units.

        Returns
        -------
        prob_v: (tf.float32) tensor of shape (n_samples, n_h) with the probability
            for the visible units to be turned on.
        h: (tf.float32) Binary tensor of shape (n_samples, n_h) representing a
            sample of the hidden units drawn from v.

        Note
        ----
        when a single vector v is passed it is important that the shape is
        (1, n_h).

        '''
        # Note shape returns a tensorflow Dimension object. We need the value
        # of that object
        batch_size = h.shape[0].value
        with tf.name_scope('sample_v_from_h'):
            prob_v = tf.nn.sigmoid(tf.matmul(h,self.weights) + self.bias_v)
            v = tf.cast(tf.random_uniform((batch_size,self.n_v)) <  prob_v,tf.float32)
        return prob_v, v

    def gibbs_step(self,X):
        '''Perform a single Gibbs step which updates first the hidden units and then
        the visible units. Can be used for reconstruction of a sample.

        Parameters
        ----------
        X: (tf.float32) Binary tensor of shape (n_samples, n_v)

        Returns
        -------
        prob_h: (tf.float32) tensor of shape (n_samples, n_h) with the
            probability for the hidden units to be turned on.
        h: (tf.float32) Binary tensor of shape (n_samples, n_h) representing a
            sample of the hidden units drawn from X.
        prob_v: (tf.float32) Tensor of shape (n_samples, n_v) with the
            probability for the visible units to be turned on given h.
        v: (tf.float32) Binary tensor of shape (n_samples, n_v) representing a
            sample of the visible units drawn from h, i.e. the reconstruction
            of X. This is the relevant output for reconstruction.

        Note
        ----
        when a single vector is passed it is important that the shape is
        (1, n_v).

        '''
        with tf.name_scope('Gibbs'):
            prob_h, h = self.sample_h_from_v(X)
            prob_v, v = self.sample_v_from_h(h)
        return prob_h, h, prob_v, v

    def fit(self,sess,data,lr,k=1,n_iter=10,batch_size=100,verbose=True):
        '''Fit the RBM to the data using k-persistent contrastive divergence (k-PCD).
        For k-CPD the Markov chain is initialized once using the training data,
        but not reset.

        Parameters
        ----------
        sess: a tf.Session() object.
        data: (np.float) Binary numpy array of shape (n_samples, n_v) containing the
            training data
        lr: the learning rate
        k: (default = 1) number of samples that are drawn from the Markov-chain
            in each step.
        n_iter: (default = 10) number of iterations. In each iteration every
            sample contained in X is visited once.
        batch_size: (default = 100) the size of the mini_batches used for the
            fitting procedure. After each mini_batch the gradient is updated.
        verbose: (default=True) print iteraton number and squared
            reconstruction error to stdout.

        Note
        ----
        1. The k-PCD algorithm implemented here is descibed in A. Fischer and
        C. Igel, "Training restricted Boltzmann machines: An Introduction",
        Pattern Recognition 47, 25 (2014).

        2. The squared reconstruction error is not necessarily a good indicator
        of whether the algorithm is truly optimizing the log-likelihood. This
        needs to be kept in mind.

        3. This function assumes that the weights and biases have already been
        initialized wit tf.variables_initializer. The reason is that this
        allows the user to run fit multiple times withot reinitializing the
        weights and biases.

        '''

        n_batch = data.shape[0]/batch_size
        X = tf.placeholder(tf.float32,shape=(batch_size,self.n_v))
        v_markov = tf.Variable(X,dtype=tf.float32,name='state_of_markov_chain')

        train_op = tf.group(self._pcd_step(X,v_markov,lr,k=k))
        error_op = self.calc_squared_reconstruction_error(X)
        # only initialze v_markove here since it is assumed that the weights and
        # biases have already been initialized.
        init = tf.variables_initializer([v_markov])

        sess.run(init,feed_dict={X:data[:batch_size,:]})
        for step in xrange(0,n_iter):
            error = 0.0
            shuffled = np.random.permutation(data)
            for j in xrange(0, n_batch):
                batch = shuffled[j*batch_size:(j+1)*batch_size]
                sess.run(train_op, feed_dict={X:batch})
                error += sess.run(error_op,feed_dict={X:batch})
            error /= n_batch

            # FOR DEBUGGING
            # mnist.plot_multiple_images(sess.run(v_markov))
            # plt.show()

            if verbose:
                print 'iteration\t{:d}\tsquared_reconstruction_error\t{:.6f}\t'.\
                    format(step, error)

    def _pcd_step(self,X,v_markov,lr,k=1):
        '''Helper function which performs a simgle step of k-PCD.

        Parameters
        ----------

        v_markov: (tf.float32) tf.Variable of shape (n_samples, n_v)
            representing the current (binary) state of the visible units in the
            Markov chain.
        For all other parameters see docstring of fit

        Returns
        -------
        Graphs for updating the weights, biases and v_markov.

        '''

        n_samples = X.shape[0].value

        with tf.name_scope('PCD'):
            prob_h_0, h_0 = self.sample_h_from_v(X)
            prob_h_k, h_k, prob_v_k, v_k = self.gibbs_step(v_markov)

            # loop below is only entered when k > 1.
            for i in xrange(1,k):
                prob_h_k, h_k, prob_v_k, v_k = self.gibbs_step(v_k)

            prob_h_k = tf.nn.sigmoid(tf.matmul(v_k,self.weights,transpose_b=True)\
                                     + self.bias_h)

            # updade weights and biases
            delta_weights = lr*tf.subtract(tf.matmul(prob_h_0,X,transpose_a=True),\
                                           tf.matmul(prob_h_k,v_k,transpose_a=True),\
                                           name='Delta_weights')/n_samples
            delta_bias_v = lr*tf.reduce_mean(X - v_k,axis=0,name='Delta_bias_v')
            delta_bias_h = lr*tf.reduce_mean(prob_h_0 - prob_h_k,axis=0,\
                                             name='Delta_bias_h')

            with tf.control_dependencies([delta_weights, delta_bias_v, delta_bias_h]):
                # control_dependencies ensures that changes in weights and biases
                # are calculated before the update is performed.
                update_weights =  tf.assign_add(self.weights,delta_weights)
                update_bias_v = tf.assign_add(self.bias_v,delta_bias_v)
                update_bias_h = tf.assign_add(self.bias_h,delta_bias_h)
                update_v = tf.assign(v_markov,v_k)

        return update_weights, update_bias_v, update_bias_h, update_v

    def calc_squared_reconstruction_error(self,X):
        '''Calculate the squared reconstruction error

        Parameters
        ----------
        X: (tf.float32) Binary tensor of shape (n_samples, n_v) containing the
            data to be reconstructed

        Returns
        -------
        squared reconstruction error (tensor)

        '''
        with tf.name_scope('reconstruction_error'):
            _, _, _, v = self.gibbs_step(X)
            error = tf.reduce_mean(tf.square(X-v))
        return error


def fit(rbm,sess,data,lr,k=1,n_iter=10,batch_size=100,verbose=True):

    n_batch = data.shape[0]/batch_size
    XX = tf.placeholder(tf.float32,shape=(batch_size,rbm.n_v))
    v = tf.Variable(XX,dtype=tf.float32,name='state_of_markov_chain')
    train_op = rbm.fit_new_2(XX,v,lr,k=k)
    error_op = rbm.calc_squared_reconstruction_error(XX)
    init = tf.global_variables_initializer()

    writer = tf.summary.FileWriter('../events/', sess.graph)
    sess.run(init,feed_dict={XX:data[:batch_size,:]})
    for i in xrange(0,n_iter):
        error = 0.0
        shuffled = np.random.permutation(data)
        for j in xrange(0, n_batch):
            batch = shuffled[j*batch_size:(j+1)*batch_size]
            sess.run(train_op, feed_dict={XX:batch})
            error += sess.run(error_op,feed_dict={XX:batch})
        if verbose:
            print i, error/n_batch
    writer.close()
    # mnist.plot_multiple_images(sess.run(rbm.weights))
    # plt.show()

if __name__ == '__main__':

    # load data and, project on twos only, and make it binary
    data, n_row, n_col = mnist.load_images('../data/train-images.idx3-ubyte')
    labels = mnist.load_labels('../data/train-labels.idx1-ubyte')
    where = labels == 2
    data = data[where,:]
    labels = labels[where]
    data /= np.amax(data)
    data = np.around(data,decimals=0)

    # plot the first 100 of the handwritten digits
    mnist.plot_multiple_images(data[:100,:])
    plt.show()

    rbm = RestrictedBoltzmannMachine(n_row*n_col,100)
    init_weights_and_biases = tf.variables_initializer([rbm.weights,\
                                                        rbm.bias_v,\
                                                        rbm.bias_h])
    # for reconstruction
    rec = tf.Variable(tf.cast(tf.random_uniform((10,n_row*n_col),maxval=2,dtype=tf.int32),\
                              tf.float32))
    init_rec = tf.variables_initializer([rec])
    update_rec = tf.assign(rec,rbm.gibbs_step(rec)[3])

    with tf.Session() as sess:
        sess.run(init_weights_and_biases)
        rbm.fit(sess,data,0.01,k=5,n_iter=20,batch_size=10)

        # Generate pictures of twos starting from a random vector
        sess.run(init_rec)
        mnist.plot_multiple_images(sess.run(rec))
        for i in xrange(0,100):
            sess.run(update_rec)
            if i %10 == 0:
                mnist.plot_multiple_images(sess.run(rec),m=10)
                plt.show()




