import numpy as np

# ===== Activation functions =====
def sigmoid(z):
    """Sigmoid activation: maps values into (0, 1)."""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation: replaces negatives with 0."""
    return np.maximum(0, z)

def softmax(z):
    """Softmax activation: converts logits into probabilities."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ===== Feedforward Neural Network Class =====
class FeedForwardNN:
    def __init__(self, layer_sizes, activation='relu', output_activation='softmax', seed=42):
        """
        Initializes a fully connected feedforward neural network.
        
        Args:
            layer_sizes (list): e.g. [784, 128, 64, 10]
            activation (str): activation function for hidden layers ('relu' or 'sigmoid')
            output_activation (str): activation for output layer ('softmax' or 'sigmoid')
            seed (int): random seed for reproducibility
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Xavier/Glorot initialization for weights
        self.params = {}
        for i in range(self.num_layers):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.params[f"W{i+1}"] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            self.params[f"b{i+1}"] = np.zeros((1, out_dim))

        # Select activation function for hidden layers
        if activation == 'relu':
            self.activation = relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            raise ValueError("Unsupported activation type")

        # Select activation function for output layer
        if output_activation == 'softmax':
            self.output_activation = softmax
        elif output_activation == 'sigmoid':
            self.output_activation = sigmoid
        else:
            raise ValueError("Unsupported output activation")

    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (ndarray): input data of shape (batch_size, input_dim)
        
        Returns:
            A (ndarray): final output (predictions)
            cache (dict): stores intermediate activations and pre-activations
        """
        cache = {'A0': X}
        A = X
        for i in range(1, self.num_layers + 1):
            Z = np.dot(A, self.params[f"W{i}"]) + self.params[f"b{i}"]
            if i < self.num_layers:
                # Hidden layer
                A = self.activation(Z)
            else:
                # Output layer
                A = self.output_activation(Z)
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A
        return A, cache

        # ======== Loss functions ========
    def compute_loss(self, y_true, y_pred, loss_type="cross_entropy", l2_lambda=0.0):
        """
        Compute the loss value.
        Args:
            y_true: one-hot encoded true labels
            y_pred: predicted probabilities
            loss_type: 'cross_entropy' or 'mse'
            l2_lambda: L2 regularization coefficient
        """
        m = y_true.shape[0]
        if loss_type == "cross_entropy":
            # Add small epsilon for numerical stability
            eps = 1e-12
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.sum(y_true * np.log(y_pred)) / m
        elif loss_type == "mse":
            loss = np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError("Unsupported loss type")

        # L2 regularization term
        l2_term = 0.0
        for i in range(1, self.num_layers + 1):
            l2_term += np.sum(np.square(self.params[f"W{i}"]))
        loss += (l2_lambda / (2 * m)) * l2_term
        return loss

    # ======== Backward propagation ========
    def backward(self, y_true, cache, l2_lambda=0.0):
        """
        Perform backpropagation to compute gradients.
        Args:
            y_true: true labels (one-hot)
            cache: dictionary from forward pass
            l2_lambda: L2 regularization coefficient
        Returns:
            grads: dictionary of gradients for W and b
        """
        grads = {}
        m = y_true.shape[0]
        A_final = cache[f"A{self.num_layers}"]

        # Compute delta for the output layer
        delta = A_final - y_true

        for i in reversed(range(1, self.num_layers + 1)):
            A_prev = cache[f"A{i-1}"]
            grads[f"dW{i}"] = (np.dot(A_prev.T, delta) / m) + (l2_lambda / m) * self.params[f"W{i}"]
            grads[f"db{i}"] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 1:
                Z_prev = cache[f"Z{i-1}"]
                if self.activation == relu:
                    dZ = (Z_prev > 0).astype(float)
                else:  # sigmoid derivative
                    s = sigmoid(Z_prev)
                    dZ = s * (1 - s)
                delta = np.dot(delta, self.params[f"W{i}"].T) * dZ

        return grads

    # ======== Parameter update ========
    def update_params(self, grads, learning_rate):
        """Update weights and biases using gradient descent."""
        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
            self.params[f"b{i}"] -= learning_rate * grads[f"db{i}"]

    # ======== Training loop ========
    def train(self, X, y, epochs=100, lr=0.01, l2_lambda=0.0, loss_type="cross_entropy", verbose=True):
        """
        Train the network using mini-batch gradient descent.
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred, cache = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred, loss_type, l2_lambda)
            
            # Backward pass
            grads = self.backward(y, cache, l2_lambda)
            
            # Parameter update
            self.update_params(grads, lr)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
