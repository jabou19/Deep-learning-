# data_loader.py
"""
Data loading and preprocessing utilities for Fashion-MNIST and CIFAR-10.
"""

import numpy as np
from keras.datasets import fashion_mnist, cifar10


def one_hot_encode(y, num_classes):
    """Convert labels (0..num_classes-1) to one-hot encoded vectors."""
    y = y.reshape(-1)
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def load_fashion_mnist(normalize=True, one_hot=True):
    """
    Load and preprocess the Fashion-MNIST dataset.
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten 28x28 -> 784
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixel values (0–255 → 0–1)
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # One-hot encode labels
    if one_hot:
        y_train = one_hot_encode(y_train, num_classes=10)
        y_test = one_hot_encode(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_cifar10(normalize=True, one_hot=True):
    """
    Load and preprocess the CIFAR-10 dataset.
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Flatten 32x32x3 -> 3072
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixel values (0–255 → 0–1)
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # One-hot encode labels
    if one_hot:
        y_train = one_hot_encode(y_train, num_classes=10)
        y_test = one_hot_encode(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


# Test the loader when running this file directly
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print("Fashion-MNIST:")
    print("Train set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)

    # Uncomment to test CIFAR-10
    # X_train, y_train, X_test, y_test = load_cifar10()
    # print("CIFAR-10:")
    # print("Train set:", X_train.shape, y_train.shape)
    # print("Test set:", X_test.shape, y_test.shape)
