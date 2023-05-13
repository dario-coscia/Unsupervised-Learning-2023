import numpy as np  # linear algebra
import matplotlib.pyplot as plt # plotting
import pandas as pd # csv manipulation
from scipy.spatial.distance import cdist # for pairwise distances kernels
from scipy.sparse import linalg as LA # for linear algebra
from scipy.sparse import csr_matrix # for sparse matrices
from sklearn.decomposition import PCA as SklearnPCA # for comparison
from sklearn.decomposition import KernelPCA as SklearnKernelPCA # for comparison
from sklearn.preprocessing import KernelCenterer # for centering


class PCA(object):
    
    def __init__(self):
        
        self.eigenvalues = None
        self.transformation_matrix = None
        self._mean = None
    
    def fit(self, x):
                
        # centered data
        self._mean = np.mean(x, axis=0)
        x_c = x - self._mean
        
        # compute covariance matrix (X^t X)/n_samples
        conv_matrix = np.dot(x_c.T, x_c) / x.shape[0]
        self.conv = conv_matrix
        
        # decompose the matrix
        eigvalues, eigvects = np.linalg.eigh(conv_matrix)
        
        # sort in ascending order
        idx = eigvalues.argsort()[::-1]   
        self.eigenvalues = eigvalues[idx]
        self.transformation_matrix = eigvects[:,idx]

        
    def transform(self, x, numb_components = None, eigen = False):
        
        # centered data
        x_c = x - self._mean
        
        # check consistency number of components
        numb_components = self._check_numb_components(x_c, numb_components)
        
        # choose the eigenvect/eigenval according to the number of components
        eigvalues = self.eigenvalues[:numb_components]
        transform_matrix = self.transformation_matrix[:, :numb_components]
        
        # project data
        projection = np.dot(x, transform_matrix)
        
        if eigen:
            return projection, eigvalues
        
        return projection
    
    def inverse_transform(self, y):
        # number of samples and features
        n_samples, n_features = y.shape
        # define tranformation matrix
        transform_matrix = self.transformation_matrix[:, :n_features]
        # return transformation + the mean vector
        return np.dot(y, transform_matrix.T) + self._mean
    
    def _check_numb_components(self, x, numb_components):
        """
        Expect input shape to be numb_sample X dimension_sample. 
        """
        # number of samples and features
        n_samples, n_features = x.shape
        
        # check consistency
        if numb_components is None:
            numb_components = n_features
            
        if numb_components < 1 or numb_components > n_features:
            raise ValueError("numb_components must be greater or equal than one"
                             " and smaller or equal to the number of features")
            
        return numb_components


class KernelPCA(object):

    def __init__(self, kernel, gamma=None):
        """Kernel PCA implementation.

        :param kernel: pca kernel {'poly', 'rbf'} possible option.
        :type kernel: string
        :param gamma: coefficient for rbf, poly kernel; ignored by others.
        :type gamma: int
        """

        # list of possible kernels
        possible_kernels = ['poly', 'gaussian']

        if kernel not in possible_kernels:
            raise ValueError(
                "expected kernel to be in list of possible kernels.")
        else:
            self._kernel = self._choose_kernel(kernel)

        # gamma coefficient + alpha coefficient
        if gamma is None:
            ValueError
        self._gamma = gamma


    def fit_transform(self, X, numb_components=None, eigen=False):

        (m, _) = X.shape

        self._check_numb_components(X, numb_components)

        # creating kernel matrix
        K = self._kernel(X)

        # centering matrix
        K = KernelCenterer().fit_transform(K)
        K = csr_matrix(K)
        
        # slow implementation
        # I = np.ones(shape=(m,m))/m
        # K  =  K - I @ K - K @ I + I @ K @ I   

        # performing svd
        eigvalues, eigvects = LA.eigsh(K, k=numb_components, which='LA')


        # # sort in ascending order
        idx = eigvalues.argsort()[::-1]
        eigvalues = eigvalues[idx]
        eigvects = eigvects[:, idx]

        # project data (use shortcut for just projecting)
        projection = eigvects * np.sqrt(eigvalues)

        if eigen:
            return projection, eigvalues

        return projection


    def _choose_kernel(self, kernel_type):
        """
        Returning the right kernel.
        """
        def rbf(X, X_train=None):
            if X_train is None:
                X_train = X
            dists = cdist(X, X_train, metric="sqeuclidean")
            K = np.exp( - dists * self._gamma)
            return K

        def poly(X, X_train=None):
            if X_train is None:
                X_train = X
            K = 1. + np.dot(X, X_train.T)
            return np.power(K, self._gamma)

        kernels = {'gaussian': rbf,
                   'poly': poly}

        return kernels[kernel_type]

    def _check_numb_components(self, x, numb_components):
        """
        Expect input shape to be numb_sample X dimension_sample. 
        """
        # number of samples and features
        _, n_features = x.shape

        # check consistency
        if numb_components is None:
            numb_components = n_features

        if numb_components < 1 or numb_components > n_features:
            raise ValueError("numb_components must be greater or equal than one"
                             " and smaller or equal to the number of features")

        return numb_components