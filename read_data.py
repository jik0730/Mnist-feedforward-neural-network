import sys
import struct
import array
import autograd.numpy as np


"""
Description: Read mnist data set.
Inputs (data sets): Rare mnist data sets.
Outputs (parsed data sets): Parsed mnist data sets appropriately.
"""
def mnist(f1, f2, f3, f4):
    def parse_labels(filename):
        with open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            temp = np.array(array.array("B", fh.read()), dtype=np.uint8)
            toReturn = np.zeros((num_data, 10))
            for i in range(num_data):
                toReturn[i,temp[i]] = 1
            return toReturn

    def parse_images(filename):
        with open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows*cols)

    train_images = parse_images(f1)
    train_labels = parse_labels(f2)
    test_images  = parse_images(f3)
    test_labels  = parse_labels(f4)

    return train_images, train_labels, test_images, test_labels
