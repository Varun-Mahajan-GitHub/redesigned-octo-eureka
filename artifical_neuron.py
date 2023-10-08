import math


def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y


def activate(inputs, weights):
    sum = 0
    for x, w in zip(inputs, weights):
        sum += x*w

    return sigmoid(sum)


if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    output = activate(inputs, weights)
    print(output)
