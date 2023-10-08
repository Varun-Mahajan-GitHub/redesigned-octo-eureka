import math
from random import random

import numpy as np


# Save activations and derivatives
# implement back propagation
# Implement gradient descent
# Implement train
# train our network with some dummy dataset
# Make some predictions


class MLP:
    """
    A multilayer perceptron class
    """

    def __init__(self, num_inputs=2, num_hidden=[3, 3], num_outputs=1):
        """
        Constructor for MLP
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Create a generic rep of the layers
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        self.weights = []
        # Create random connection weight for the layers
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propogate(self, inputs):

        activations = inputs
        self.activations[0] = inputs
        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)
            # calculate activation
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error, verbose=False):
        """
        dE/dW_i = (y- a_[i+1]) s'(h_[i+1]) a_i
        s'(h_[i+1]) = s(h_[i+1])(1- s(h_[i+1]))
        s(h_[i+1]) = a_[i+1]

        dE/dW_[i-1] = (y- a_[i+1]) s'(h_[i+1])) W_i s'(h_i)a[i-1]
        :param error:
        :return:
        """

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(1, delta.shape[0])
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print(f'The weight matrix {i} is {self.derivatives[i]}')
        return error

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for (input, target) in zip(inputs, targets):
                # create forward propgation
                output = self.forward_propogate(input)
                # print(f'The prediction by the model is {output}')
                # calculate_error
                error = target - output

                # back propagate
                self.back_propagate(error)

                # Apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            # report error
            print(f'Error: {sum_error / len(inputs)} at epoch {i}')

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _mse(self, target, output):
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    # Create a MLP
    mlp = MLP(2,[5],  1)
    # Train our mlp
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    mlp.train(inputs,targets,50,0.2)

    input = np.array([.1, .3])

    output = mlp.forward_propogate(input)
    print()
    print()
    print(f'Our network believes {input[0]} + {input[1]} is {output[0]}')