import sys
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from read_data import mnist
from functions import softmax
from functions import batch
from functions import regression
from functions import cross_entropy
from functions import accuracy


def main():
    # Load data set
    print("Load training data...")
    training_images, training_labels, test_images, test_labels = mnist()

    # Initialize parameters (Hidden layer: 1)
    batch_size = 100
    learning_rate = 2
    hl_units = 113
    W1 = np.ones((785,hl_units)) # W1 appended by bias(b1)
    W2 = np.ones((hl_units, 10)) # W2 appended by bias(b2)

    # Gradient functions
    grad_W1 = grad(cross_entropy, argnum=0)
    grad_W2 = grad(cross_entropy, argnum=1)

    # For plotting
    plt_y = []
    
    # Training
    print("Start training...")
    for i in range(1000):
        batch_xs, batch_ys = batch(training_images, training_labels)
        batch_xs = np.append(batch_xs, np.ones((batch_size,1)), axis=1)
        plt_y.append(cross_entropy(W1, W2, batch_xs, batch_ys))
        W2 -= grad_W2(W1, W2, batch_xs, batch_ys) * learning_rate
        W1 -= grad_W1(W1, W2, batch_xs, batch_ys) * learning_rate

    plt.plot(plt_y)
    plt.show()

    # Print accuracy
    accuracy(W1, W2, test_images, test_labels)


if __name__ == '__main__':
    main()
