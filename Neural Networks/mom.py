import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from nn import NeuralNetwork

from sklearn.decomposition import PCA
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
flat_images = data[range(len(data.columns) - 1)].as_matrix()
print 'Training with ', len(flat_images), 'samples\n'

# Calculate image moments
X = []
for img in flat_images:
    img = img.reshape((28, 28))  # reshape array into 2d image
    img = np.array(img, np.uint8)  # convert to uint8

    mom = np.array(cv2.moments(img).values())
    X.append(mom)

X = np.array(X)

X = scale(X)  # scale data

# Perform PCA on data to reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot the data in 2d
A = data[[len(data.columns) - 1]].as_matrix()
zoo = []
colormap = {0: 'gray', 1: 'blue', 2: 'red', 3: 'green',
            4: 'purple', 5: 'pink', 6: 'orange', 7: 'brown',
            8: 'yellow', 9: 'black'}
for i, xr in enumerate(X_reduced):
    if (i % 10) != 0:
        continue
    x, y = xr
    zoo.append((x, y, colormap[A[i][0]], A[i]))
for x, y, c, l in zoo:
    i += 1
    plt.scatter(x, y, c=c, label=l)

# Extract targets
y = data[[len(data.columns) - 1]].as_matrix()
y = to_categorical(y)

# Declare our model
model = NeuralNetwork(g=Sigmoid, dg=SigmoidGradient,
                      L=CrossEntropy, dL=CrossEntropyGradient)

# Add layers
model.addLayer(24)  # add layer of 24 neurons for 24 image moments
model.addLayer(50)  # add layer of 50 neurons
model.addLayer(10)  # we add one neuron for each class (0-9)

# Train the model
model.train(X, y, max_it=200)

# Get the accuracy of n random entries

n = 1000  # number of images to sample
max_images = 3   # number of image to plot
acc = 0.  # accuracy
images_plotted = 0
for i in range(n):
    index = np.random.randint(len(X))  # get random entry
    pred = np.argmax(model.predict(X[index]))  # get prediction of said entry
    # get the target (aka what should be the prediction)
    target = np.argmax(y[index])

    print 'Predict X[', index, ']:', np.argmax(model.predict(X[index]))
    print 'Real value y[', index, ']:', np.argmax(y[index])
    print

    # plot the missclassified images
    if pred == target:
        acc += 1e0
    else:
        if max_images > images_plotted:
            img = flat_images[index].reshape((28, 28))
            plt.figure()
            title = str(target) + ' missclassified as ' + str(pred)
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
