from layer import Layer

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer: Layer):
        """
            Adds a Layer to the model.
        """
        self.layers.append(layer)

    def use_loss(self, loss, loss_prime):
        """
            Sets the loss function and its derivative.
        """
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        """
            Returns a prediction for a given input.
        """
        result = []

        for i in range(len(input_data)):
            # forward pass
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_pass(output)

            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate):
        """
            Trains the network with the given data x_train, y_train.
            
            epochs: Number of iterations of the learning process.

            learning_rate: Adjustment factor for backpropagation.
        """
        for i in range(epochs):
            err = 0

            for j in range(len(x_train)):
                # forward pass
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_pass(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward pass
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)

            # average error
            err /= len(x_train)
            print("epoch %d/%d, error = %f" % (i+1, epochs, err))
