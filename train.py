# train.py
"""
Train a simple Feedforward Neural Network (FFNN) from scratch on Fashion-MNIST.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_fashion_mnist
from ffnn import FeedForwardNN


def accuracy(y_true, y_pred):
    """Compute accuracy between one-hot labels and predicted probabilities."""
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)


def main():
    # ===== 1️⃣ Load and preprocess data =====
    print("Loading Fashion-MNIST dataset...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    # Take a smaller subset for faster training during testing
    X_train_small, y_train_small = X_train[:5000], y_train[:5000]
    X_test_small, y_test_small = X_test[:1000], y_test[:1000]

    # ===== 2️⃣ Initialize model =====
    model = FeedForwardNN(
        layer_sizes=[784, 128, 64, 10],
        activation='relu',
        output_activation='softmax',
    )

    # ===== 3️⃣ Train the model =====
    epochs = 50
    lr = 0.1
    print(f"Training for {epochs} epochs with learning rate={lr}...")
    losses = []

    for epoch in range(epochs):
        y_pred, cache = model.forward(X_train_small)
        loss = model.compute_loss(y_train_small, y_pred)
        grads = model.backward(y_train_small, cache)
        model.update_params(grads, lr)

        acc = accuracy(y_train_small, y_pred)
        losses.append(loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    # ===== 4️⃣ Evaluate on test data =====
    y_pred_test, _ = model.forward(X_test_small)
    test_acc = accuracy(y_test_small, y_pred_test)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

    # ===== 5️⃣ Plot loss curve =====
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
