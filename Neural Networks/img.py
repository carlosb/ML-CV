import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from nn import NeuralNetwork
from nn import to_categorical

from sklearn.preprocessing import scale


def CrossEntropy(y_hat, y):
    """
    Consult Wikipedia for the brief explanation of
    cross entropy loss. It has a nice example for
    logistic regression which is basically what w
    are trying to achieve.

    https://en.wikipedia.org/wiki/Cross_entropy
    """
    N = len(y)
    CE = np.sum((y * np.log(y_hat)) + ((1. - y) * np.log(1. - y_hat)))
    CE *= -1. / float(N)
    return CE


def CrossEntropyGradient(y_hat, y):
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
    return np.multiply(Sigmoid(x), (1. - Sigmoid(x)))


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


# Read data
data = pd.read_csv('mnist.txt', sep=" ", header=None)

# Extract input
X = data[range(len(data.columns) - 1)].as_matrix()
print 'Training with ', len(X), 'samples\n'

# Extract targets
y = data[[len(data.columns) - 1]].as_matrix()
y = to_categorical(y)

# Declare our model
model = NeuralNetwork(g=Sigmoid, dg=SigmoidGradient, L=CrossEntropy,
                      dL=CrossEntropyGradient)

# Add layers
model.addLayer(784)  # each vector consists of 784 entries
model.addLayer(20)  # add layer of 20 neurons
model.addLayer(20)  # add layer of 20 neurons
model.addLayer(10)  # add layer of 10 neurons (for 10 digit classes)

# Scale data
X = scale(X)

# Train the model
model.train(X, y, max_it=200, epsilon_init=0.2)

# Get the accuracy of n random entries
n = 100  # NUMBER OF ENTRIES TO SAMPLE
max_images = 3

acc = 0.
images_plotted = 0
for i in range(n):
    index = np.random.randint(len(X))  # get random entry
    pred = np.argmax(model.predict(X[index]))  # get prediction of said entry
    target = np.argmax(y[index])  # get the target

    print 'Predict X[', index, ']:', np.argmax(model.predict(X[index]))
    print 'Real value y[', index, ']:', np.argmax(y[index])
    print

    # plot the missclassified images
    if pred == target:
        acc += 1.
    else:
        if max_images > images_plotted:
            img = X[index].reshape((28, 28))
            plt.figure()
            title = target, ' missclassified as ', pred
            plt.title(title)
            plt.imshow(img, interpolation='nearest')
            images_plotted += 1
acc /= n
print 'Accuracy', acc * 100, '%'


# Plot the graph
plt.figure()
plt.title('Plot of the network\'s cost over time')
plt.xlabel('t')
plt.ylabel('cost')
plt.plot(model.cost_history)
plt.show()
