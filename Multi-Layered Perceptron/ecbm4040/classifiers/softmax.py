import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    for i in range(X.shape[0]):
        
        soft = X[i].dot(W)
        Sum = 0
        
        for j in range(soft.shape[0]):
            
            Sum += np.exp(soft[j])
            
        soft = np.exp(soft)/Sum
        
        dW_y = (soft - (y[i]==np.arange(10)).astype(int))
        
        dW += (X[i].reshape(X[i].shape[0],1)).dot(dW_y.reshape(1,dW_y.shape[0]))
        loss += -np.log(soft[y[i]])
        
    loss /= X.shape[0]
    dW /= X.shape[0]
    
    loss += reg * np.sum(W * W)
    dW += reg*2*W
                
        

            

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    soft = np.exp(X.dot(W))
    Sum_mat = np.repeat(np.sum(soft, axis = 1).reshape(soft.shape[0],1), soft.shape[1],axis = 1)
    norm_soft = soft/Sum_mat
    dW = X.T.dot(norm_soft - (y[:,None]==np.arange(soft.shape[1])).astype(int))
    
    dW /= X.shape[0]
    loss = np.mean(-np.log(norm_soft[np.arange(norm_soft.shape[0]), y]))
    
    loss += reg * np.sum(W * W)
    dW += reg*2*W
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
