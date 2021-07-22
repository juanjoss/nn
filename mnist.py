import numpy as np

from network import Network
from layer import FCLayer, ActivationLayer
from activation import tanh, tanh_prime
from loss import MSE, MSE_prime

from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as np_utils

# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data: 60.000 samples

# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

# encode output
y_train = np_utils.to_categorical(y_train)

# test data: 10.000 samples

# reshape and normalize test data
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255

# encode output
y_test = np_utils.to_categorical(y_test)

# network
nn = Network()
nn.add(FCLayer(28*28, 100))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(FCLayer(100, 50))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(FCLayer(50, 10))
nn.add(ActivationLayer(tanh, tanh_prime))

# train
nn.use_loss(MSE, MSE_prime)
nn.train(x_train[0:1000], y_train[0:1000], epochs=50, learning_rate=0.1)

# test
out = nn.predict(x_test[0:3])
print("\npredicted values: ", out)
print("\ntrue values: ", y_test[0:3])
