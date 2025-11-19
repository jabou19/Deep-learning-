"""
Define and train a simple feed-forward neural network using only NumPy.

Features:
- Configurable: num_hidden_layers, n_hidden_units, learning_rate, batch_size,
  l2_coeff, weights_init, activation (relu/tanh/sigmoid), loss (mse/cross_entropy),
  optimizer (currently: "sgd").
- Implements forward pass, backward pass, mini-batch gradient descent,
  and evaluation (accuracy, loss curves, confusion matrix).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ============================================================
# Utilities
# ============================================================

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    y = y.astype(int)
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute classification accuracy from probabilities/logits and labels."""
    ## accuracy calculation between predicted and target labels
    y_pred = np.argmax(pred, axis=1)
    return float((y_pred == target).mean())


# ============================================================
# Activation functions
# ============================================================
# Exercise 2 f) Add  activation functions and its derivatives
class Activation:
    def __init__(self, name: str):
        name = name.lower()
        if name not in {"relu", "tanh", "sigmoid"}:
            raise ValueError(f"Unsupported activation: {name}")
        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.name == "relu":
            return np.maximum(0, x)
        if self.name == "tanh":
            return np.tanh(x)
            #return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  #TODO : for ckecking code 
        if self.name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        raise RuntimeError
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        if self.name == "relu":
           # return (x > 0).astype(np.float32)
            return np.where(x > 0, 1.0, 0.0).astype(np.float32) #TODO : for checking 
        if self.name == "tanh":
            t = np.tanh(x)
            #t= (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  #TODO : for ckecking code
            return 1.0 - t ** 2
        if self.name == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        raise RuntimeError


# ============================================================
# Loss functions
# ============================================================
## Exercise 2 k) Implement cross entropy  
# # and MSE loss functions
class Loss:
    def __init__(self, name: str):
        name = name.lower()
        if name not in {"mse", "cross_entropy"}:
            raise ValueError(f"Unsupported loss: {name}")
        self.name = name

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # MES loss is the mean of squared differences between predicted and true values
        # # y_true is one-hot targets,  y_pred is probabilities
        if self.name == "mse":
            return float(np.mean((y_true - y_pred) ** 2))
        # Cross-entropy loss measures how different two probability distributions are: the true labels vs the model’s predicted probabilities.
        if self.name == "cross_entropy":
            eps = 1e-12
            y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
            return float(-np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1)))
        raise RuntimeError

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.name == "mse":
            return 2.0 * (y_pred - y_true) / y_pred.shape[0]
        """
        ombine softmax + cross entropy and differentiate with respect to the logits 
       z_k (the inputs to softmax), the result simplifies to:
        dL/dz_k = (y_pred_k - y_true_k)
        """
        if self.name == "cross_entropy":
            return (y_pred - y_true) / y_pred.shape[0]
        raise RuntimeError


# ============================================================
# Weight initialization
# ============================================================
## Exercise 2  i) Glorot and He initialization
def init_weights(shape: Tuple[int, int], method: str) -> np.ndarray:
    """Initialize weights with / Xavier / He / small normal."""
    method = method.lower()
    fan_in, fan_out = shape
    # Hidden layers with tanh / sigmoid: use Xavier (Glorot)
    if method == "xavier": #TODO depending on activation function
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape).astype(np.float32)
    # Hidden layers with ReLU, use He initialization
    if method == "he": #TODO depending on activation function
        # scaling a standard normal by sqrt(2/fan_in)
        return (np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)).astype(np.float32)

    # default: small normal
    return (0.01 * np.random.randn(fan_in, fan_out)).astype(np.float32)


# ============================================================
# Feed-Forward Neural Network (FFNN) Exercise 2+3+4
# ============================================================


class FFNN:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_hidden_layers: int,
        n_hidden_units: int,
        activation: str = "r",
        loss: str = "c",
        learning_rate: float = 0.0,
        l2_coeff: float = 0.0,
        weights_init: str = "h",
        optimizer: str = "adam",
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.act = Activation(activation)
        self.loss_fn = Loss(loss)
        self.lr = learning_rate #lr: learning rate
        self.l2 = l2_coeff # L2 regularization coefficient
        self.weights_init = weights_init
       
        optimizer = optimizer.lower()
        if optimizer not in {"adam"}:
         raise ValueError(f"Unsupported optimizer: {optimizer}")
        self.optimizer = optimizer

        # Adam optimizer state (Adam) // exercise 3.4 +4.1
        self.t = 0
        self.mW: List[np.ndarray] = [] # mw is standing for first moment estimate for weights
        self.vW: List[np.ndarray] = [] # vw is standing for second moment estimate for weights
        self.mb: List[np.ndarray] = [] # mb is standing for first moment estimate for biases
        self.vb: List[np.ndarray] = [] # vb is standing for second moment estimate for biases
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8 # eps is a small constant to prevent division by zero

        # List of weight matrices and biases
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        # Input -> hidden layers
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            W = init_weights((prev_dim, n_hidden_units), weights_init)
            b = np.zeros((1, n_hidden_units), dtype=np.float32)
            self.W.append(W)
            self.b.append(b)
            # Adam state for this layer
            self.mW.append(np.zeros_like(W))
            self.vW.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b))
            self.vb.append(np.zeros_like(b))
            prev_dim = n_hidden_units

        # Last hidden -> output
        W_out = init_weights((prev_dim, num_classes), weights_init)
        b_out = np.zeros((1, num_classes), dtype=np.float32)
        self.W.append(W_out)
        self.b.append(b_out)
        self.mW.append(np.zeros_like(W_out))
        self.vW.append(np.zeros_like(W_out))
        self.mb.append(np.zeros_like(b_out))
        self.vb.append(np.zeros_like(b_out))

    @staticmethod
    # lecture 1 page 36 softmax for classification
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    # lecture 2 page 14 forward pass 17 
    """Forward pass; returns (pre_activations, activations)."""
    def forward(self, x: np.ndarray):
        a = x
        pre_activations: List[np.ndarray] = []
        activations: List[np.ndarray] = [x]

        # Hidden layers
        for i in range(self.num_hidden_layers):
            z = a @ self.W[i] + self.b[i]
            a = self.act(z)
            pre_activations.append(z)
            activations.append(a)

        # Output layer
        z_out = a @ self.W[-1] + self.b[-1]
        pre_activations.append(z_out)
        y_out = self._softmax(z_out)
        activations.append(y_out)

        return pre_activations, activations
    
    """Loss +  L2 regularization."""
    def compute_loss(self, y_pred: np.ndarray, y_true_oh: np.ndarray) -> float:
       
        base = self.loss_fn(y_pred, y_true_oh)
        # l2: is regularization coefficient
        if self.l2 > 0.0:
            reg = sum(np.sum(W ** 2) for W in self.W)
            base += self.l2 * reg / (2.0 * y_true_oh.shape[0])
        return base
    # lecture 2 page 17 backward pass
    def backward(self, pre_activations, activations, y_true_oh):
        """Backward pass; returns gradients (dW, db)."""
        dW = [np.zeros_like(W) for W in self.W]
        db = [np.zeros_like(b) for b in self.b]

        # Output layer gradient (softmax + cross-entropy)
        y_pred = activations[-1]
        delta = (y_pred - y_true_oh) / y_true_oh.shape[0]  # (N, C)

        # Last layer
        a_prev = activations[-2]
        dW[-1] = a_prev.T @ delta + self.l2 * self.W[-1]
        db[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers backwards
        for i in range(self.num_hidden_layers - 1, -1, -1):
            z = pre_activations[i]
            da = delta @ self.W[i + 1].T
            dz = da * self.act.derivative(z)
            a_prev = activations[i]
            dW[i] = a_prev.T @ dz + self.l2 * self.W[i]
            db[i] = np.sum(dz, axis=0, keepdims=True)
            delta = dz

        return dW, db
    """Optimizer step (Adam ).""" # // exercise 3.4 +4.1
    # exercise 3.1 +4
    def step(self, dW, db):

        self.t += 1
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        # lr: learning rate 
        lr_t = self.lr * np.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t)
        for i in range(len(self.W)):
            # Update first and second moments for weights
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr_t * mW_hat / (np.sqrt(vW_hat) + eps)

            # Update first and second moments for biases
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.b[i] -= lr_t * mb_hat / (np.sqrt(vb_hat) + eps)
    # Predict probabilities 
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, acts = self.forward(x)
        return acts[-1]
    # Predict classes labels
    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)


# ============================================================
# Training helpers
# ============================================================


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
):
    """Yield mini-batches of (X_batch, y_batch)."""
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        if end > N:
            break
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train_ffnn(
    model: FFNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    num_epochs: int = 0,
    batch_size: int = 0,
    validation_every_steps: int = 0,
):
    """Train FFNN with mini-batch gradient descent."""
    num_classes = model.num_classes
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    step = 0
    # this loop for epoch in range num_epochs
    for epoch in range(num_epochs):
        train_losses = []
        train_accs = []
        #  this loop for mini-batches 
        for inputs, targets in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs_flat = inputs.reshape(inputs.shape[0], -1)
            targets_oh = one_hot(targets, num_classes)
            # Forward pass: get prediction and loss
            pre_acts, acts = model.forward(inputs_flat)
            y_pred = acts[-1] # [-1] is output layer
            loss_val = model.compute_loss(y_pred, targets_oh)

            dW, db = model.backward(pre_acts, acts, targets_oh)
            model.step(dW, db)

            train_losses.append(loss_val)
            train_accs.append(accuracy(y_pred, targets))
        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        # Validation
        if X_valid is not None and y_valid is not None and step % validation_every_steps == 0:
            Xv_flat = X_valid.reshape(X_valid.shape[0], -1)
            yv_oh = one_hot(y_valid, num_classes)
            # Forward pass
            yv_pred = model.predict_proba(Xv_flat)
            val_loss = model.compute_loss(yv_pred, yv_oh)
            val_acc = accuracy(yv_pred, y_valid)
        else:
            val_loss = np.nan
            val_acc = np.nan

        history["valid_loss"].append(float(val_loss))
        history["valid_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch+1:03d} |.... "
            f"train loss: {avg_train_loss:.4f}, train acc: {avg_train_acc:.4f} | "
            f"valid loss: {val_loss:.4f}, valid acc: {val_acc:.4f}"
        )

    return history
