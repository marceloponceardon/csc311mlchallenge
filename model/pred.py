#!/bin/python3
'''
Module docstring:

Instructions:
Submit a python3 script that takes as parameter the name of a CSV file containing the test set,
and produces predictions for that test set.
'''

# The script can use the following libraries
# As well as basic python imports like sys, csv, random
# You will NOT be able to use sklearn, pytorch, tensorflow, etc. in the final submitted code.
# You may reuse any code that you wrote or was provided to you from the labs.
# We should be able to make ~60 predictions within 1 minute (1 prediction per second)

from importlib.metadata import version
import sys
# import csv
# import random
# import numpy as np
import pandas as pd

def main():
    '''
    TODO:
    Main method, reads in the test data,
    calls relevant functions to make predictions,
    and writes to a csv file
    '''
    # Print version of python, and import libraries using importlib
    print("Pandas version: ", version('pandas'))
    print("Numpy version: ", version('numpy'))

    # Check if the correct number of arguments are passed
    if len(sys.argv) != 2:
        print("Usage: python3 pred.py <test_file>")
        sys.exit(1)
    # Read in the test data
    test_data = pd.read_csv(sys.argv[1])
    print(test_data)

def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`
    """
    if len(z.shape) == 1:
        max = np.max(z)
        exp = np.exp(z - max)
        result = exp / np.sum(exp)
    else:
        max = np.max(z, axis=1, keepdims=True)
        exp = np.exp(z - max)
        result = exp / np.sum(exp, axis=1, keepdims=True)

    return result

def make_onehot(indicies, total=128):
    """
    Convert indicies into one-hot vectors by
    first creating an identity matrix of shape [total, total],
    then indexing the appropriate columns of that identity matrix.

    Parameters:
        `indices` - a numpy array of some shape where
                    the value in these arrays should correspond to category
                    indices (e.g. note values between 0-127)
        `total` - the total number of categories (e.g. total number of notes)

    Returns: a numpy array of one-hot vectors
        If the `indices` array is shaped (N,)
           then the returned array will be shaped (N, total)
        If the `indices` array is shaped (N, D)
           then the returned array will be shaped (N, D, total)
        ... and so on.
    """
    I = np.eye(total)
    return I[indicies]

def do_forward_pass(model, X):
    """
    Compute the forward pass to produce prediction for inputs.

    This function also keeps some of the intermediate values in
    the neural network computation, to make computing gradients easier.

    For the ReLU activation, you may find the function `np.maximum` helpful

    Parameters:
        `model` - An instance of the class MLPModel
        `X` - A numpy array of shape (N, model.num_features)

    Returns: A numpy array of predictions of shape (N, model.num_classes)
    """
    model.N = X.shape[0]
    model.X = X
    model.m = np.dot(X, model.W1) + model.b1
    model.h = np.maximum(model.m, 0)
    model.z = np.dot(model.h, model.W2) + model.b2
    model.y = softmax(model.z)
    return model.y


def do_backward_pass(model, ts):
    """
    Compute the backward pass, given the ground-truth, one-hot targets.

    You may assume that `model.forward()` has been called for the
    corresponding input `X`, so that the quantities computed in the
    `forward()` method is accessible.

    The member variables you store here will be used in the `update()`
    method. Check that the shapes match what you wrote in Part 2.

    Parameters:
        `model` - An instance of the class MLPModel
        `ts` - A numpy array of shape (N, model.num_classes)
    """
    model.z_bar = (model.y - ts) / model.N
    model.W2_bar = np.dot(model.h.T, model.z_bar)
    model.b2_bar = np.sum(model.z_bar, axis=0)
    model.h_bar = np.dot(model.z_bar, model.W2.T)
    model.m_bar = model.h_bar * (model.m > 0)
    model.W1_bar = np.dot(model.X.T, model.m_bar)
    model.b1_bar = np.sum(model.m_bar, axis=0)


class MLPModel(object):
    def __init__(self, num_features=128*20, num_hidden=100, num_classes=128):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # weights and biases for the first layer of the MLP
        # note that by convention, the weight matrix shape has the
        # num input features along the first axis, and num output
        # features along the second axis
        self.W1 = np.zeros([num_features, num_hidden])
        self.b1 = np.zeros([num_hidden])

        # weights and biases for the second layer of the MLP
        self.W2 = np.zeros([num_hidden, num_classes])
        self.b2 = np.zeros([num_classes])

        # initialize the weights and biases
        self.initializeParams()

        # set all values of intermediate variables (to be used in the
        # forward/backward passes) to None
        self.cleanup()

    def initializeParams(self):
        """
        Initialize the weights and biases of this two-layer MLP to be random.
        This random initialization is necessary to break the symmetry in the
        gradient descent update for our hidden weights and biases. If all our
        weights were initialized to the same value, then their gradients will
        all be the same!
        """
        self.W1 = np.random.normal(0, 2/self.num_features, self.W1.shape)
        self.b1 = np.random.normal(0, 2/self.num_features, self.b1.shape)
        self.W2 = np.random.normal(0, 2/self.num_hidden, self.W2.shape)
        self.b2 = np.random.normal(0, 2/self.num_hidden, self.b2.shape)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """
        return do_forward_pass(self, X) # To be implemented below

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return do_backward_pass(self, ts) # To be implemented below

    def loss(self, ts):
        """
        Compute the average cross-entropy loss, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters of this model.

        Parameters:
            `alpha` - A number representing the learning rate
        """
        self.W1 = self.W1 - alpha * self.W1_bar
        self.b1 = self.b1 - alpha * self.b1_bar
        self.W2 = self.W2 - alpha * self.W2_bar
        self.b2 = self.b2 - alpha * self.b2_bar

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        # To be filled in during the forward pass
        self.N = None # Number of data points in the batch
        self.X = None # The input matrix
        self.m = None # Pre-activation value of the hidden state, should have shape
        self.h = None # Post-RELU value of the hidden state
        self.z = None # The logit scores (pre-activation output values)
        self.y = None # Class probabilities (post-activation)
        # To be filled in during the backward pass
        self.z_bar = None # The error signal for self.z2
        self.W2_bar = None # The error signal for self.W2
        self.b2_bar = None # The error signal for self.b2
        self.h_bar = None  # The error signal for self.h
        self.m_bar = None # The error signal for self.z1
        self.W1_bar = None # The error signal for self.W1
        self.b1_bar = None # The error signal for self.b1


if __name__ == "__main__":
    main()
