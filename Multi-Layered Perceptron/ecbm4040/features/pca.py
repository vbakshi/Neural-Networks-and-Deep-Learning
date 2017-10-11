import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    
    X_cov_eig, X_cov_vec = np.linalg.eig(X.T.dot(X))
    P = (X_cov_vec[:,np.argsort(X_cov_eig)[-1:-K-1:-1]]).T
    T = X_cov_eig[np.argsort(X_cov_eig)[-1:-K-1:-1]]
    


    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
