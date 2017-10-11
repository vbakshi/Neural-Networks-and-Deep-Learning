from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    out = x.reshape(x.shape[0],w.shape[0]).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: input data, of shape (N, d_1, ... d_k)
      - w: weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis = 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x,0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    drel = (x>0).astype(int)
    dx = drel*dout

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - X: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    reg = 0.000005
    dx = np.zeros_like(x)
    #############################################################################
    # TODO: You can use the previous softmax loss function here.                #
    #############################################################################
    
    soft = np.exp(x - np.max(x,axis = 1).reshape(-1,1))
    Sum_mat = np.sum(soft, axis = 1).reshape(-1,1)
    norm_soft = soft/Sum_mat
    
    loss = np.mean(-(x - np.max(x,axis = 1).reshape(-1,1))[np.arange(x.shape[0]),y] + np.log(Sum_mat))

    
    dx = norm_soft - (y[:,None]==np.arange(soft.shape[1])).astype(int)
    dx = dx / x.shape[0]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dx