"""
Authors:
 Carlos Brito (carlos.brito524@gmail.com)
 Laura Alonzo ()

This is a small implementation of a Fully Connected Multi
Layer Perceptron. Currently, it's set to train over the
MNIST dataset.

To run make sure you have mnist.txt in the same folder.
After which you only need to run it with:

~> $ python nn.py


Notes:
    - It is being trained using the whole of the database
    - We are training the method with the Conjugate Gradient Method
    - The cost function is called Cross Entropy
    - >>>> It will plot the missclassified images <<<<<
    - It will plot a graph of the cost function as time passes
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, g, dg, L, dL, reg=1):
        """
        Parameters
        ----------
        l : int
            Number of layers.
        g : function
            Activation function.
        dg : function
            Derivative of activation
        L : function
            Loss function.
        dL : function
            Gradient of loss function.
        """

        # number of layers
        self.layers = 0

        # activation function
        self.activation = g

        # activation gradient
        self.activation_gradient = dg

        # loss model
        self.loss = L

        # gradient of loss
        self.loss_gradient = dL

        # number of nodes per each layer
        self.layer_sizes = []

        # current epoch
        self.epoch = 0

        # reg param
        self.reg_param = reg

        # training data
        self.X = None

        # targets
        self.y = None

        # weight tensor
        self.W = None

        # list of vectors for linear combination at each layer
        self.a = []

        # activation vectors of each layer
        self.h = []

        # list of gradients of the weigths at each layer
        self.weight_gradient = []

        # cost list
        self.cost_history = []

    def addLayer(self, size):
        """
        Parameters
        ----------
        size : int
            Number of neurons in layer.

        Example
        -------
        If we want a layer of 3 nodes then we call
        ````
        model.addLayer(3)
        ````
        """
        self.layer_sizes.append(size)
        self.layers += 1

    def train(self, X, y, max_it=200, epsilon_init=0.5):
        """
        Trains the method by minimizing the loss function and
        initializing the weights. It takes the whole dataset
        and the respective targets. It utilizes the Conjugate
        Gradient method to perform the minimization.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        y : array_like
            Targets.
        max_it : int
            Max iterations.
        epsilon_init : float
            Weight initialization parameter.
        """
        from scipy import optimize

        self.X = X
        self.y = y
        self.W = self.init_weights(epsilon_init)
        self.epoch = 0

        params = self.getParams()
        options = {'maxiter': max_it, 'disp': True}
        _res = optimize.minimize(self.objective, params, jac=True,
                                 method='CG', args=(X, y), options=options,
                                 callback=self.callbackF)

        self.setParams(_res.x)

    def predict(self, x):
        """
        Predict a new input x by forward
        propagating the network.

        Parameters
        ----------
        x : array_like
            New input.

        Returns
        -------
        prediction : array_like or float
            Total cost of network (forward-propagation)
        """
        prediction = self.forward(x, self.y)
        return prediction

    def init_weights(self, epsilon):
        """
        Initializes the weights randomly aiding itself
        from the shapes of adjacent layers.

        Parameters
        ----------
        epsilon : float
            Low and high bound of interval.
        """
        W = np.array([
            np.random.rand(self.layer_sizes[i - 1],
                           self.layer_sizes[i]) * 2. * epsilon - epsilon
            for i in range(1, self.layers)
        ])
        return W

    def forward(self, X, y):
        """
        Forward propagation of the network.

        Parameters
        ----------
        X : array_like
            Input sample.
        y : array_like
            Target.
        """
        h_old = X
        self.h = []
        self.a = []
        for k in range(self.layers - 1):
            a = 1 + np.dot(h_old, self.W[k])
            h_new = self.activation(a)

            self.h.append(h_old)  # these two lines just memorize the vectors
            self.a.append(a)

            h_old = h_new
        y_hat = h_new
        self.h.append(y_hat)

        self.y_hat = y_hat

        return y_hat

    def back_prop(self, y_hat):
        """
        Algorithm to calculate the gradients of the network
        Taken from Deep Learning by Ian Goodfellow, Yoshua Bengio
        and Aaron Courville.

        This method saves the gradients of the weights in self.weight_gradient
        so you can use them for later in any method such as SGD or
        Batch Training.

        Parameters
        ----------
        y_hat : array_like
            The prediction for an input X. We only need this value to get
        the error at the last layer.
        """
        g = self.loss_gradient(y_hat, self.y)  # ERROR: THIS SHOULD BE CHANGED TO self.loss_gradient(y_hat, self.y)
        self.weight_gradient = []
        for k in reversed(range(self.layers - 1)):
            g = np.multiply(g, self.activation_gradient(self.a[k]))  # deltas

            weight_gradient = (1. / float(len(y_hat))) * np.dot(self.h[k].T, g) \
                + self.reg_param / len(self.y) * self.W[k]

            self.weight_gradient.insert(0, weight_gradient)

            g = np.dot(g, self.W[k].T)  # propagate gradients to lower layer

        return self.weight_gradient

    def objective(self, params, X, y):
        """
        Objective function that acts as a way of obtaining
        the cost of the network on input X and gets the gradients.

        Parameters
        ----------
        params : array of array_like elements
            All the weights concatenated into a vector.
        X : array_like
            Input sample.
        y : array_like
            Targets.
        """
        self.setParams(params)
        y_hat = self.forward(X, y)
        cost = self.loss(y_hat, y) \
            + (self.reg_param / (2 * len(self.y))) \
            * np.sum(self.getParams()**2)

        self.back_prop(y_hat)
        grad = self.getWeightGradients(X, y)
        return cost, grad

    def callbackF(self, params):
        """
        We use this to shuffle the training set at each epoch and do some
        other interactivity stuff.
        """
        self.setParams(params)

        cost, grad = self.objective(params, self.X, self.y)

        self.cost_history.append(cost)

        print 'epoch:', self.epoch
        print 'Cost:', cost

        self.epoch += 1

    def getWeightGradients(self, X, y):
        """
        Concatenates and returns all the calculated weight gradients
        """
        grads = np.array([])
        for w in self.weight_gradient:
            grads = np.concatenate((grads, w.ravel()))
        return grads

    def getParams(self):
        """
        Get all the weights concatenated into one single 1 dimensional vector
        """
        params = np.array([])
        for w in self.W:
            params = np.concatenate((params, w.ravel()))
        return params

    def setParams(self, params):
        """
        Reconstruct the parameters given by a unidimensional vector.
        Note we use the fact that we know the size of each layer
        """
        w_start = 0
        w_end = 0
        for i in range(1, self.layers):
            w_start = w_end
            w_end = w_start + self.layer_sizes[i - 1] * self.layer_sizes[i]
            self.W[i - 1] = np.reshape(params[w_start:w_end],
                                       (self.layer_sizes[i - 1],
                                        self.layer_sizes[i]))


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
