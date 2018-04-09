import numpy as np
import matplotlib.pyplot as plt
import struct

def load_images(fname):
    '''load the MNIST image data

    Parameters
    ----------
    fname: name of file containing the images in the original idx format

    Returns
    -------
    X: numpy array containng the images. Each row corresponds to one image.
    n_row: The number of rows of one image.
    n_col: The number of columns of one image.

    '''
    f = open(fname,'rb')
    _ = f.read(4) # first 4 bytes are not needed
    n_image = struct.unpack('>i',f.read(4))[0] # > is for big endian, i is for integer (4 bytes)
    n_row = struct.unpack('>i',f.read(4))[0]
    n_col = struct.unpack('>i',f.read(4))[0]

    X = np.zeros((n_image,n_row*n_col))
    for i in xrange(0,n_image):
        for j in xrange(0,n_row*n_col):
            pix = struct.unpack('>B',f.read(1))[0]
            X[i,j] = pix
    return X, n_row, n_col

def load_labels(fname):
    '''load the MNIST labels

    Parameters
    ----------
    fname: name of file containing the labels in the original idx format

    Returns
    -------
    y: numpy array containng the labels.

    '''
    f = open(fname)
    _ = f.read(4) # first 4 bytes are not needed
    n_labels = struct.unpack('>i',f.read(4))[0]

    y = np.zeros(n_labels)
    for i in xrange(0,n_labels):
        label  = struct.unpack('>B',f.read(1))[0]
        y[i] = label
    return y

def plot_image(X,image_index,n_row=28,n_col=28):
    '''Plot a single image from X.

    Parameters
    ----------
    X: numpy array containig the images. Each row contains one image.
    n_row: The number of rows of one image.
    n_col: The number of columns of one image.

    Returns
    -------
    fig: pyplot figure instance

    '''
    fig = plt.figure(figsize=(8,8))
    x = X[image_index,:]
    x = np.reshape(x,(n_row,n_col))
    plt.imshow(x,cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    return fig

def plot_multiple_images(X_subset,n_row=28,n_col=28,m=10):
    '''Plot the images contained in X_subset on a grid with m columns and n rows.

    Parameters
    ----------
    X_subset: numpy array containig some of the images. Each row contains one
       image.
    n_row: The number of rows of one image.
    n_col: The number of columns of one image.

    Returns
    -------
    fig: pyplot figure instance
    '''
    n = (X_subset.shape[0]+m-1) / m
    fig = plt.figure(figsize=(m,n))
    for i in range(0,X_subset.shape[0]):
        plt.subplot(n,m,i+1)
        x = X_subset[i,:]
        x = np.reshape(x,(n_row,n_col))
        plt.imshow(x,cmap='Greys')
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    return fig

def crop_images(X,left=4,righ=4,up=4,down=4):
    '''Crop the images. Still Needs to be finished.

    '''
    # NEED TO FINISH THIS
    X_crop = X[:,up:down]
    X_crop = X_crop.T
    X_crop = X_crop[left:]

if __name__ == '__main__':
    X = load_images('./t10k-images.idx3-ubyte')
    y = load_labels('./t10k-labels.idx1-ubyte')
    print X.shape
    X = X/np.amax(X)
    plot_image(X,1210)
    print y[1210]
    X_s = X[np.random.randint(0,X.shape[0],27)]
    f = plot_multiple_images(X_s)
    g = plot_multiple_images(X_s)
    plt.show()





