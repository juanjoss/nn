import numpy as np

from network import Network
from layer import FCLayer, ActivationLayer
from activation import tanh, tanh_prime
from loss import MSE, MSE_prime

# train data
x_train = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]],
])

# labels
y_train = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]],
])

# network

# For XOR at least 2 neurons are needed in order to solve the problem.
# Solving XOR with one hidden layer with 2 neurons:

nn = Network()
nn.add(FCLayer(2, 2))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(FCLayer(2, 1))
nn.add(ActivationLayer(tanh, tanh_prime))

# train
nn.use_loss(MSE, MSE_prime)
nn.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = nn.predict(x_train)
print(out)
