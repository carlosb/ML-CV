"""
Author: Carlos Brito (carlos.brito524@gmail.com)
        Laura Alonzo

Date: 15/05/2017

Description
-----------
This scripts demos the use of the kernel trick.

I.e. It generates concentric circles and then
utilizes the kernel trick to classify both classes
with a linear classifier.

Usage
-----
python 1.py

Output
------
Plots dataset and prints accuracy of classifier

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles


def phi(x):
    """
    The feature space mapping.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.array([x1, x2, np.power(x1, 2) + np.power(x2, 2)])


def k(x0, x1):
    """
    The kernel function.
    """
    return np.dot(phi(x0).T, phi(x1))


# Generate concentric circles
X1, Y1 = make_circles(n_samples=800, noise=0.1, factor=0.2)
Y1[np.where(Y1 == 0)] = -1.  # change label from 0 to -1

# Plot circles
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()

# Set up parameters for training
N = len(X1)  # number of samples
lamb = 1  # lambda
K = k(X1, X1)  # kernel matrix
Id = np.identity(N)  # identity matrix
t = Y1  # targets

# Minimize
a = np.linalg.inv(K + lamb * Id)
a = np.dot(a, t)

# Check accuracy
sample_count = 400
total = 0.
for i in range(sample_count):
    idx = np.random.randint(0, N)

    # Predict
    x = np.array([X1[idx]])
    y = np.dot(k(x, X1), a) + 0.5

    if np.sign(y) == t[idx]:
        total += 1.

acc = total / sample_count

print 'Accuracy: ', acc * 100, '%'
