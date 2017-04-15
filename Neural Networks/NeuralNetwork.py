"""
Authors:
 Carlos Brito (carlos.brito524@gmail.com)
 Laura Alonzo ()

This is a small implementation of a Fully Connected Multi
Layer Perceptron. Currently, it's set to train over the
MNIST dataset.

To run make sure you have mnist.txt in the same folder.
After which you only need to run it with 

~> $ python NeuralNetwork.py


Notes:
	- It is being trained using the whole of the database
	- We are training the method with the Conjugate Gradient Method
	- The cost function is called Cross Entropy
	- >>>> It will plot the missclassified images <<<< PLEASE 
	- It will plot a graph of the cost function as time passes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

Costs = []

def CrossEntropy(y_hat, y):
	"""
	Consult Wikipedia for the brief explanation of
	cross entropy loss. It has a nice example for
	logistic regression which is basically what w
	are trying to achieve.

	https://en.wikipedia.org/wiki/Cross_entropy
	"""
	N = len(y)
	CE = np.sum( ( y * np.log(y_hat) ) + ( (1.-y) * np.log(1.-y_hat) ) )
	CE *= -1./float(N)
	return CE

def CrossEntropyGradient(y_hat,y):
	"""
	Gradient of Cross Entropy with respect to
	the weights.
	"""
	return y_hat - y

def Sigmoid(x):
	"""
	Sigmoid function.
	"""
	return 1. / (1. + np.exp(-x))

def SigmoidGradient(x):
	"""
	Gradient of sigmoid function.
	"""
	return np.multiply(Sigmoid(x),(1. - Sigmoid(x)))


class NeuralNetwork:
	def __init__(self, g, dg, L, dL, reg=1):
		"""
		@var l - number of layers
		@var g - activation
		@var dg - derivative of activation
		@var L - loss
		@var dL - gradient of loss
		"""

		## number of layers
		self.layers = 0

		## activation function
		self.activation = g

		## activation gradient
		self.activation_gradient = dg

		## loss model
		self.loss = L

		## gradient of loss
		self.loss_gradient = dL

		## number of nodes per each layer
		self.layer_sizes = []

		## current epoch
		self.epoch = 0

		## reg param
		self.reg_param = reg

		## training data
		self.X = None

		## targets
		self.y = None

		## weight tensor
		self.W = None

		## list of vectors for linear combination at each layer
		self.a = []

		## activation vectors of each layer
		self.h = []

		## list of gradients of the weigths at each layer
		self.weight_gradient = []

	def addLayer(self, size):
		"""
		@param size is the size of the layer i.e.
		if we want a layer of 3 nodes then we
		call model.addLayer(3)
		"""
		self.layer_sizes.append(size)
		self.layers += 1

	def train(self,X,y, max_it=200, epsilon_init=0.5):
		"""
		Trains the network using an optimizer.
		Please select the method to be used in the call to optimize.minimize(method=YourChoice)
		Also feel free to adjust the weight initialization parameter.
		"""

		self.X = X
		self.y = y
		self.W = self.init_weights(epsilon_init)
		self.epoch = 0

		params = self.getParams()
		options = {'maxiter': max_it, 'disp' : True}
		_res = optimize.minimize(self.objective, params, jac=True, method='CG', \
                                 args=(X, y), options=options, callback=self.callbackF)

		self.setParams(_res.x)

	def predict(self,x):
		"""
		Predict a new input x by forward
		propagating the network
		"""
		prediction = self.forward(x,self.y)
		return prediction


	def init_weights(self,epsilon):
		"""
		initializes the weights randomly using
		the shapes of adjacent layers
		"""
		W = np.array([

			np.random.rand(self.layer_sizes[i-1], self.layer_sizes[i]) * 2. * epsilon - epsilon
			for i in range(1,self.layers)

			])
		return W

	def forward(self,X,y):
		"""
		Forward propagation of the network
		X - input matrix
		y - output vector
		"""
		h_old = X
		self.h = []
		self.a = []
		for k in range(self.layers-1):
			a = 1 + np.dot( h_old, self.W[k])
			h_new = self.activation( a )

			self.h.append(h_old) # these two lines just memorize the vectors
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
		so you can use them for later in any method such as SGD or Batch Training

		y_hat - The prediction for an input X. We only need this value to get
		the error at the last layer.
		"""
		g = self.loss_gradient(y_hat,y)
		self.weight_gradient = []
		for k in reversed(range(self.layers-1)):
			g = np.multiply(g, self.activation_gradient(self.a[k])) # deltas

			weight_gradient = (1./float(len(y_hat)))*np.dot(self.h[k].T, g) + self.reg_param/len(self.y) * self.W[k] # calculate weight gradients

			self.weight_gradient.insert(0,weight_gradient)

			g = np.dot(g, self.W[k].T) # propagate gradients to lower layer

		return self.weight_gradient

	def objective(self,params,X,y):
		"""
		Objective function that acts as a way of obtaining
		the cost of the network on input X and gets the gradients.

		params - basically all the weights concatenated into a vector
		X - input
		y - targets
		"""
		self.setParams(params)
		y_hat = self.forward(X,y)
		cost = model.loss(y_hat,y) + (self.reg_param/(2*len(self.y)))*np.sum(self.getParams()**2)
		model.back_prop(y_hat)
		grad = self.getWeightGradients(X,y)
		return cost, grad

	def callbackF(self, params):
		"""
		Helper function for the optimizer to do something at each iteration
		
		We use this to shuffle the training set at each epoch and do some
		other interactivity stuff.

		"""
		self.setParams(params)

		randomize = np.arange(len(self.X))
		np.random.shuffle(randomize)
		self.X = self.X[randomize]
		self.y = self.y[randomize]

		cost, grad = self.objective(params, self.X, self.y)

		Costs.append(cost)

		print 'epoch:', self.epoch
		print 'Cost:', cost

		self.epoch += 1

	def getWeightGradients(self,X,y):
		"""
		Concatenates and returns all the calculated weight gradients
		"""
		grads = np.array([])
		for w in self.weight_gradient:
			grads = np.concatenate(( grads, w.ravel() ))
		return grads

	def getParams(self):
		"""
		Get all the weights concatenated into one single 1 dimensional vector
		"""
		params = np.array([])
		for w in self.W:
			params = np.concatenate(( params, w.ravel() ))
		return params

	def setParams(self,params):
		"""
		Reconstruct the parameters given by a unidimensional vector.
		Note we use the fact that we know the size of each layer
		"""
		w_start = 0
		w_end = 0
		for i in range(1,self.layers):
			w_start = w_end
			w_end = w_start + self.layer_sizes[i-1]*self.layer_sizes[i]
			self.W[i-1] = np.reshape(params[w_start:w_end], (self.layer_sizes[i-1], self.layer_sizes[i]))

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


# <<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN PROGRAM >>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Read data
data = pd.read_csv('mnist.txt', sep=" ", header=None)

# Extract input
X = data[range(len(data.columns)-1)].as_matrix()
print 'Training with ', len(X), 'samples\n'

# Extract targets
y = data[[len(data.columns)-1]].as_matrix()
y = to_categorical(y)

# Declare our model
model = NeuralNetwork(g=Sigmoid, dg=SigmoidGradient, L=CrossEntropy, dL=CrossEntropyGradient)

# Add layers
model.addLayer(784) # each vector consists of 784 entries
model.addLayer(20) # each vector consists of 784 entries
model.addLayer(20) # each vector consists of 784 entries
model.addLayer(10) # we add one node for each class (0-9)

# Train the model
model.train(X,y,max_it=200)

# Get the accuracy of n random entries
n = 100			# NUMBER OF ENTRIES TO SAMPLE
max_images = 3 	# <<<<<<<<<<<<<<
				# SET AS 0 IF YOU DONT WISH TO PLOT THE MISSCLASIFICATIONS
			 	# OTHERWISE, INTERPRET AS THE MAX NUMBER OF MISSCLASIFCATIONS
			 	# THAT WILL BE PLOTTED 

acc = 0.	 	
images_plotted = 0
for i in range(n):
	index = np.random.randint(len(X)) # get random entry
	pred = np.argmax(model.predict(X[index])) # get prediction of said entry
	target = np.argmax(y[index]) # get the target (aka what should be the prediction)

	print 'Predict X[',index,']:', np.argmax(model.predict(X[index]))
	print 'Real value y[', index,']:', np.argmax(y[index])
	print

	# plot the missclassified images
	if pred == target:
		acc += 1.
	else:
		if max_images > images_plotted:
			img = X[index].reshape((28,28))
			plt.figure()
			title = target, ' missclassified as ',pred
			plt.title(title)
			plt.imshow(img, interpolation='nearest')
			images_plotted += 1
acc /= n
print 'Accuracy', acc*100, '%'


# Plot the graph
plt.figure()
plt.title('Plot of the network\'s cost over time')
plt.xlabel('t')
plt.ylabel('cost')
plt.plot(Costs)
plt.show()

