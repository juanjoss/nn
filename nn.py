import numpy as np
from utils import step

class Layer():
    def __init__(self, neurons=1, input_shape=2, act_fun=step):
        self.weights = np.zeros((neurons, input_shape))
        self.b = np.zeros((input_shape, 1))
        self.act_fun = act_fun
    
    def activate(self, x: np.array):
        print("weights: ", self.weights)
        print("b: ", self.b)
        
        y = np.dot(self.weights, x) + self.b
        output = self.act_fun(y, 0)
        print("output: ", output)

        return output

class NeuralNetwork():
    def __init__(self, layers=1):
        self.layers = []
        self.step = step

        print("creating {} layers...".format(layers))
        for i in range(layers):
            self.layers.append(Layer(neurons=1, input_shape=2, act_fun=self.step))

    def __forward_pass(self, input_data=[]):
        for layer in self.layers:
            layer_output = layer.activate(input_data)
            print("layer output: ", layer_output)

    def train(self, training_data=[], epochs=100, learning_rate=0.001):
        for epoch in range(epochs):
            print("epoch {}...".format(epoch))

            for i in range(len(training_data)):
                print("\nforward pass for: ", training_data[i])
                self.__forward_pass(training_data[i])

    def loss(self):
        pass

if __name__ == "__main__":
    # AND neuron test
    
    AND_test = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    nn = NeuralNetwork()
    nn.train(AND_test, epochs=1)
