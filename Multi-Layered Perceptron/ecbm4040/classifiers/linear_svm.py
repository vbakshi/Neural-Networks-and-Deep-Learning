import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Multi-class Linear SVM loss function, naive implementation (with loops).
    
    In default, delta is 1 and there is no penalty term wst delta in objective function.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D, C) containing weights.
    - X: a numpy array of shape (N, D) containing N samples.
    - y: a numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns:
    - loss: a float scalar
    - gradient: wrt weights W, an array of same shape as W
    """
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0    
    
    for i in range(num_train):
        scores = X[i].dot(W)

        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Linear SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero
    

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_classes = W.shape[1]
    num_train = X.shape[0]


    scores = X.dot(W)
    correct_class_score = np.repeat(scores[np.arange(len(scores)), y].reshape(X.shape[0],1), 10, axis = 1)
    ind = (y[:,None]!=np.arange(10)).astype(int)
    margin = scores - correct_class_score + ind
    loss = np.sum(margin*((margin>0).astype(int)))
    

    Sum_mat = (np.sum((margin>0).astype(int), axis = 1)).reshape(X.shape[0],1)
    temp = Sum_mat*(y[:,None]==np.arange(10)).astype(int)
    dW = ((X.T).dot((margin>0).astype(int) - temp))
    
    loss /= num_train
    dW /= num_train
    
    dW += reg*2*W
    loss += reg * np.sum(W * W)

    
    
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
    
