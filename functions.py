import autograd.numpy as np
import random
import sys
from autograd import grad


"""
Description: Compute softmax funciton.
Input_1 (np.array(10,?)): Linear equation(Wx + b).
Output_1 (np.array(10,?)): Output of softmax(y).
"""
def softmax(x):
    max = np.amax(x, axis=0)
    z = x - max
    return np.exp(z) / np.sum(np.exp(z), axis=0)


"""
Description: Regression function.
Input_1 (np.array(?,785)): Batch input.
Input_2 (np.array(785,10)): Parameter W.
Output_1 (np.array(?,10)): Expected distribution of y.
"""
def regression(x, W):
    return np.transpose(softmax(np.transpose(np.matmul(x, W))))


"""
Description: Output batch of size 100.
Input_1 (np.array(?,784)): Input images data set.
Input_1 (np.array(?,10)): Input labels data set.
Output_1 (np.array(100,784), np.array(100, 10)): Batch of size 100.

Suggestion: Maybe need to prevent replicates of random rows.
"""
def batch(img, lab):
    toReturn1 = np.empty((0,784))
    toReturn2 = np.empty((0,10))
    for i in range(100):
        r = random.randint(1, img.shape[0]-1)
        toReturn1 = np.append(toReturn1, [img[r]], axis=0)
        toReturn2 = np.append(toReturn2, [lab[r]], axis=0)
    return toReturn1, toReturn2


"""
Description: ReLU function.
Input (np.array(batch_size, hl_units)): z_1
Output (np.array(batch_size, h1_units)): a_1(z_1 applied by ReLU)
"""
def ReLU(z1):
    return np.maximum(z1, 0)


"""
Description: Cross entropy loss function (objective function).
Input_1 (np.array(785,?)): W1 weight matrix between Input and Hidden.
Input_2 (np.array(?,10)): W2 weight matrix between Hidden and Output.
Input_3 (np.array(batch_size,785)): Minibatch of input.
Input_4 (np.array(batch_size,10)): Corresponding true target value of minibatch.
Output_1 (float): Cross-entropy scalar value.
"""
def cross_entropy(W1, W2, X, Y):
    z1 = np.matmul(X, W1)
    a1 = ReLU(z1)
    z2 = np.matmul(a1, W2)
    y_ = np.transpose(softmax(np.transpose(z2)))
    y_ += sys.float_info.min
    toReturn = np.mean(-np.sum(Y*np.log(y_), axis=1))
    return toReturn


"""
Description: Print accuracy of the neural network.
"""
def accuracy(W1, W2, XT, YT):
    y_ = np.argmax(YT, axis=1)
    XT = np.append(XT, np.ones((10000,1)), axis=1)
    z1 = np.matmul(XT, W1)
    a1 = ReLU(z1)
    z2 = np.matmul(a1, W2)
    y = np.argmax(np.transpose(softmax(np.transpose(z2))), axis=1)
    correct = np.mean(np.equal(y_, y))
    print("Correctness: ", correct)


"""
Description: Forward propagation. (NOT COMPLETED)
"""
def backward(W1, W2, XT, YT):
    z1 = np.matmul(X, W1)
    a1 = ReLU(z1)
    z2 = np.matmul(a1, W2)
    y_ = np.transpose(softmax(np.transpose(z2)))
    y_ += sys.float_info.min
    def cross_ent(Y_, Y):
        return np.mean(-np.sum(Y*np.log(Y_), axis=1))
    grad_c = grad(cross_ent, argnum=0)



