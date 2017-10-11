from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        velocity = []
        for i in range(len(layers)):
            velocity.extend([0,0])
        
        self.velocity = velocity
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        #TODO: Feedforward                                #
        ###################################################
        
        out = X
        for layer in self.layers:
            out = layer.feedforward(out)
        
        loss, dout = softmax_loss(out, y)
        
        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        
        dX = dout
        
        for layer in self.layers[::-1]:
            
            dX = layer.backward(dX)
        
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        
        square_weights = np.sum([np.sum(layer.params[0]**2) for layer in layers])
        loss += 0.5*self.reg*square_weights
                
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        # creates new lists with all parameters and gradients
        params = []
        grads = []
        layers = self.layers
        num_layers = self.num_layers
        for  layer in layers:
            params.extend(layer.params)
            grads.extend(layer.gradients)
                    

        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layers                              #
        ###################################################
        gamma = 0.5
        
        for i in range(len(params)): 
            self.velocity[i] = gamma*self.velocity[i] + learning_rate*grads[i]
            params[i] -= self.velocity[i]
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        
        out  = X
        for layer in layers:
            
            out = layer.feedforward(out)
            

        
        soft = np.exp(out)/np.sum(np.exp(out),axis = 1).reshape(out.shape[0],1)
        predictions = np.argmax(soft,axis = 1)
        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


