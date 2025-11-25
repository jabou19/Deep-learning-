"""
Define and train a simple feed-forward neural network using only NumPy.

Features Configurable:
- Num epochs, num_hidden_layers, n_hidden_units, learning_rate, batch_size,
  l2_coeff, weights_init, activation (relu/tanh/sigmoid), loss (mse/cross_entropy),
  optimizer (currently: "Adam").
- Implements forward pass, backward pass, mini-batch gradient descent,
  and evaluation (accuracy curves, loss curves, confusion matrix).
- Implementation stages: 
  •	Forward pass: matrix multiplications + activation functions
  •	Loss computation: MSE or cross-entropy with L2 regularization
  •	Backward pass: manual derivative calculation and weight updates
  •	Training loop: mini-batch gradient descent
  •	Evaluation: compute accuracy, loss curves, and confusion matrices

"""

import importlib
import copy

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Mapping

_WANDB_MODULE: Optional[Any] = None

def _ensure_wandb() -> Optional[Any]:
    """Dynamically import wandb if it is available in the environment."""
    global _WANDB_MODULE
    if _WANDB_MODULE is not None:
        return _WANDB_MODULE
    try:
        module = importlib.import_module("wandb")
    except ImportError:
        return None
    _WANDB_MODULE = module
    return module

# ============================================================
# Utilities
# ============================================================
 # convert integer labels to one-hot vectors
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    y = y.astype(int)
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

# compute accuracy between predicted and target labels
def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute classification accuracy from probabilities/logits and labels."""
    y_pred = np.argmax(pred, axis=1)
    return float((y_pred == target).mean())

#TODO compute histograms of parameters per layer
def compute_parameter_histograms(
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    bins: int = 20,
) -> List[dict]:
    """Summarize per-layer parameter distributions for basic diagnostics."""
    histograms: List[dict] = []
    for W, b in zip(weights, biases):
        layer_params = np.concatenate((W.ravel(), b.ravel()))
        counts, bin_edges = np.histogram(layer_params, bins=bins)
        histograms.append({"counts": counts, "bin_edges": bin_edges})
    return histograms

# TODO compute gradient norms per layer
def compute_layer_gradient_norms(dW: List[np.ndarray], db: List[np.ndarray]) -> List[float]:
    """Return L2 norms of gradients per layer (weights + biases)."""
    norms: List[float] = []
    for grad_W, grad_b in zip(dW, db):
        # Combine gradients here so spikes in either weights or biases surface.
        layer_grad = np.concatenate((grad_W.ravel(), grad_b.ravel()))
        norms.append(float(np.linalg.norm(layer_grad)))
    return norms


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
    # Activation function (relu/tanh/sigmoid)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.name == "relu":
            return np.maximum(0, x)
        if self.name == "tanh":
            return np.tanh(x)
            #return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  #TODO : for ckecking code 
        if self.name == "sigmoid":
            return self._sigmoid(x)
        raise RuntimeError
    # Derivative of activation function (relu/tanh/sigmoid)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        if self.name == "relu":
           # return (x > 0).astype(np.float32)
            return np.where(x > 0, 1.0, 0.0).astype(np.float32) #TODO : for checking 
        if self.name == "tanh":
            t = np.tanh(x)
            #t= (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  #TODO : for ckecking code
            return 1.0 - t ** 2
        if self.name == "sigmoid":
            s = self._sigmoid(x)
            return s * (1.0 - s)
        raise RuntimeError

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid implementation."""
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)
        out = np.empty_like(x)
        positive = x >= 0
        if np.any(positive):
            out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        negative = ~positive
        if np.any(negative):
            exp_x = np.exp(x[negative])
            out[negative] = exp_x / (1.0 + exp_x)
        return out


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
    #TODOreturn (0.01 * np.random.randn(fan_in, fan_out)).astype(np.float32)


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
        weights_init: str = "he",
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
        requested_init = str(weights_init).lower()
        init_aliases = {
            "h": "he",
            "he": "he",
            "x": "xavier",
            "glorot": "xavier",
            "xavier": "xavier",
        }
        requested_init = init_aliases.get(requested_init, requested_init)
        if requested_init not in {"he", "xavier"}:
            raise ValueError("weights_init must be either 'he' or 'xavier'.")
        if self.act.name == "relu" and requested_init != "he":
            raise ValueError("Use He initialization with ReLU hidden layers.")
        if self.act.name in {"tanh", "sigmoid"} and requested_init != "xavier":
            raise ValueError("Use Xavier initialization with tanh or sigmoid hidden layers.")
        self.weights_init = requested_init
       
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
            W = init_weights((prev_dim, n_hidden_units), self.weights_init)
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
        W_out = init_weights((prev_dim, num_classes), self.weights_init)
        b_out = np.zeros((1, num_classes), dtype=np.float32)
        self.W.append(W_out)
        self.b.append(b_out)
        self.mW.append(np.zeros_like(W_out))
        self.vW.append(np.zeros_like(W_out))
        self.mb.append(np.zeros_like(b_out))
        self.vb.append(np.zeros_like(b_out))

    @staticmethod
    # lecture 1 page 36 softmax for classification
    # Softmax function to convert logits to probabilities
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the main hyperparameters."""
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "num_hidden_layers": self.num_hidden_layers,
            "n_hidden_units": self.n_hidden_units,
            "activation": self.act.name,
            "loss": self.loss_fn.name,
            "learning_rate": self.lr,
            "l2_coeff": self.l2,
            "weights_init": self.weights_init,
            "optimizer": self.optimizer,
        }

    # lecture 2 page 14 forward pass 17 
    """Forward pass; returns (pre_activations, activations)."""
    def forward(self, x: np.ndarray):
        a = x
        pre_activations: List[np.ndarray] = []
        activations: List[np.ndarray] = [x]

        # Hidden layers
        for i in range(self.num_hidden_layers):
            z = a @ self.W[i] + self.b[i] # z is pre-activation
            a = self.act(z) # act is activation function
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
    """Backward pass; returns gradients (dW, db)."""
    def backward(self, pre_activations, activations, y_true_oh):
        dW = [np.zeros_like(W) for W in self.W]
        db = [np.zeros_like(b) for b in self.b]

        # Output layer gradient (softmax + cross-entropy)
        y_pred = activations[-1]
        delta = (y_pred - y_true_oh) / y_true_oh.shape[0]  # (N, C)

        # Last layer
        a_prev = activations[-2]
        dW[-1] = a_prev.T @ delta + self.l2 * self.W[-1]
        db[-1] = np.sum(delta, axis=0, keepdims=True)

            # Hidden layers backward
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
            safe_vw = np.sqrt(np.maximum(vW_hat, 0.0)) + eps
            self.W[i] -= lr_t * mW_hat / safe_vw

            # Update first and second moments for biases
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            safe_vb = np.sqrt(np.maximum(vb_hat, 0.0)) + eps
            self.b[i] -= lr_t * mb_hat / safe_vb
    # Predict probabilities 
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, acts = self.forward(x) ## [input, hidden_1, …, hidden_L, output]
        return acts[-1] # -1 return output layer
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
    if batch_size <= 0:
        batch_size = N
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = indices[start:end]
        if batch_idx.size == 0:
            continue
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
    wandb_run: Optional[Any] = None,
):
    """Train FFNN with mini-batch gradient descent."""
    num_classes = model.num_classes
    print("number of classes",num_classes)
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "grad_norms": [], # per-layer gradient norms
        "param_histograms": [], # per-layer parameter histograms
    }
    wandb_module = _ensure_wandb() if wandb_run is not None else None
    if wandb_run is not None and wandb_module is None:
        raise RuntimeError("wandb_run supplied but wandb package is unavailable.")
    if wandb_run is not None:
        # Persist key hyperparameters when they are not already set (avoids sweep locks).
        existing_config = getattr(wandb_run, "config", {})
        for key, value in model.get_config().items():
            if key not in existing_config:
                existing_config[key] = value
        # Combined label for W&B charts: (weights_init + activation_function)
        try:
            existing_config["label_W_A"] = f"{model.weights_init} + {model.act.name}"
        except Exception:
            # Fail silently to avoid any CLI output or training disruption.
            pass
    step = 0
    # this loop for epoch in range num_epochs
    for epoch in range(num_epochs):
        train_losses = []
        train_accs = []
        epoch_grad_norms = [0.0 for _ in model.W] # per-layer gradient norms
        epoch_grad_steps = 0 # number of gradient updates in this epoch
        #  this loop over mini-batches 
        for inputs, targets in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs_flat = inputs.reshape(inputs.shape[0], -1)
            targets_oh = one_hot(targets, num_classes)
            # Forward pass: get prediction and loss
            pre_acts, acts = model.forward(inputs_flat)
            y_pred = acts[-1] # [-1] is output layer
            loss_val = model.compute_loss(y_pred, targets_oh)

            dW, db = model.backward(pre_acts, acts, targets_oh)
            model.step(dW, db)

            # TODOTrack gradient magnitudes to spot pathological training behaviour.
            batch_norms = compute_layer_gradient_norms(dW, db)
            for idx, norm_val in enumerate(batch_norms):
                epoch_grad_norms[idx] += norm_val
            epoch_grad_steps += 1

            train_losses.append(loss_val)
            train_accs.append(accuracy(y_pred, targets))
            step += 1
        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        # Validation (always once per epoch when validation data is supplied)
        val_loss = np.nan
        val_acc = np.nan
        validation_step = step
        if X_valid is not None and y_valid is not None:
            val_losses = []
            val_correct = 0
            val_count = 0
            val_batch_size = batch_size if batch_size > 0 else y_valid.shape[0]
            # validation loop over mini-batches
            for Xv_batch, yv_batch in iterate_minibatches(X_valid, y_valid, val_batch_size, shuffle=False):
                Xv_flat = Xv_batch.reshape(Xv_batch.shape[0], -1)
                yv_oh = one_hot(yv_batch, num_classes)
                # Forward pass for validation
                yv_pred = model.predict_proba(Xv_flat)
                # Compute validation loss and accuracy
                val_losses.append(model.compute_loss(yv_pred, yv_oh))
                # Compute number of correct predictions
                val_correct += (np.argmax(yv_pred, axis=1) == yv_batch).sum()
                val_count += yv_batch.size
            val_loss = float(np.mean(val_losses)) if val_losses else np.nan
            val_acc = float(val_correct / val_count) if val_count else np.nan

        history["valid_loss"].append(float(val_loss))
        history["valid_acc"].append(float(val_acc))
        if epoch_grad_steps:
            avg_grad_norms = [val / epoch_grad_steps for val in epoch_grad_norms]
        else:
            avg_grad_norms = [np.nan for _ in epoch_grad_norms]
            #TODO 
        history["grad_norms"].append(avg_grad_norms)
        # Parameter histograms help confirm weights stay well-distributed.
        histograms = compute_parameter_histograms(model.W, model.b)
        history["param_histograms"].append(histograms)

        if wandb_run is not None and wandb_module is not None:
            log_payload: Dict[str, Any] = {
                "epoch": epoch,
                "global_step": step,
                "metrics/train_loss": avg_train_loss,
                "metrics/train_accuracy": avg_train_acc,
                "metrics/valid_loss": val_loss,
                "metrics/valid_accuracy": val_acc,
            }
            for layer_idx, norm in enumerate(avg_grad_norms):
                log_payload[f"grad_norms/layer_{layer_idx}"] = norm
            for layer_idx, (W, b) in enumerate(zip(model.W, model.b)):
                layer_params = np.concatenate((W.ravel(), b.ravel()))
                log_payload[f"params/layer_{layer_idx}"] = wandb_module.Histogram(layer_params)

            wandb_run.log(log_payload, step=step)

        print(
            f"Epoch {epoch:02d} | validation step {validation_step:04d} | "
            f"train loss: {avg_train_loss:.4f}, train acc: {avg_train_acc:.4f} | "
            f"valid loss: {val_loss:.4f}, valid acc: {val_acc:.4f}"
        )

    if wandb_run is not None and wandb_module is not None:
        # Populate summary metrics for quick leaderboard comparisons.
        best_valid_acc = np.nan
        if history["valid_acc"]:
            valid_acc_array = np.asarray(history["valid_acc"], dtype=np.float32)
            if not np.all(np.isnan(valid_acc_array)):
                best_valid_acc = float(np.nanmax(valid_acc_array))
        summary_payload: Dict[str, Any] = {
            "summary/final_train_loss": history["train_loss"][-1] if history["train_loss"] else np.nan,
            "summary/final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else np.nan,
            "summary/final_valid_loss": history["valid_loss"][-1] if history["valid_loss"] else np.nan,
            "summary/final_valid_accuracy": history["valid_acc"][-1] if history["valid_acc"] else np.nan,
            "summary/best_valid_accuracy": best_valid_acc,
        }
        if hasattr(wandb_run, "config"):
            activation_name = str(wandb_run.config.get("activation", "") or "").strip()
            weights_init_name = str(wandb_run.config.get("weights_init", "") or "").strip()
            if activation_name:
                summary_payload["summary/activation"] = activation_name
            if weights_init_name:
                summary_payload["summary/weights_init"] = weights_init_name
        wandb_run.summary.update(summary_payload)

    return history


def build_wandb_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """Return example sweep configurations for random and Bayesian runs."""
    metric = {"name": "metrics/valid_accuracy", "goal": "maximize"}
    parameters = {
        "num_hidden_layers": {"values": [2]},
        "n_hidden_units": {"values": [256]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [100]},
        "l2_coeff": {"values": [1e-3]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [15]},
    }
    random_cfg = {
        "method": "random",
        "metric": metric,
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "min_iter": 5},
    }
    bayes_cfg = {
        "method": "bayes",
        "metric": metric,
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "min_iter": 5},
    }
    grid_cfg = {
        "method": "grid",
        "metric": metric,
        "parameters": parameters,
    }
    return {"random": random_cfg, "Bayesian": bayes_cfg, "grid": grid_cfg}


def _infer_input_dim(data: np.ndarray) -> int:
    """Infer flattened input dimensionality from training data."""
    if data.ndim < 2:
        raise ValueError("Expected `data` with shape (N, ...) for input inference.")
    return int(np.prod(data.shape[1:]))


def _infer_num_classes(y_train: np.ndarray, y_valid: Optional[np.ndarray]) -> int:
    """Infer class count assuming zero-based integer labels."""
    classes = np.unique(y_train)
    if classes.size == 0:
        raise ValueError("`y_train` must contain at least one label to infer classes.")
    if y_valid is not None:
        classes = np.union1d(classes, np.unique(y_valid))
    return int(classes.max() + 1)


def _deep_update_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-updated copy of `base` using `overrides`."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update_dict(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_activation_init_from_config(config: Mapping[str, Any]) -> Tuple[str, str]:
    """Return activation and initializer while enforcing explicit pairing."""
    activation = str(config.get("activation", "relu")).strip().lower()
    if activation not in {"relu", "sigmoid", "tanh"}:
        raise ValueError(f"Unsupported activation '{activation}' in sweep config.")

    raw_init = config.get("weights_init")
    if raw_init is None or str(raw_init).strip() == "":
        weights_init = "he" if activation == "relu" else "xavier"
    else:
        weights_init = str(raw_init).strip().lower()

    init_aliases = {
        "h": "he",
        "he": "he",
        "x": "xavier",
        "glorot": "xavier",
        "xavier": "xavier",
    }
    weights_init = init_aliases.get(weights_init, weights_init)

    if weights_init not in {"he", "xavier"}:
        raise ValueError("weights_init must be 'he' or 'xavier'.")
    if activation == "relu" and weights_init != "he":
        raise ValueError("ReLU hidden layers must use He initialization.")
    if activation in {"sigmoid", "tanh"} and weights_init != "xavier":
        raise ValueError("Sigmoid/tanh hidden layers must use Xavier initialization.")

    return activation, weights_init


def run_wandb_sweep(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    project: str,
    entity: Optional[str] = None,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    class_names: Optional[np.ndarray] = None,
    log_artifacts: bool = True,
    sweep_name: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
    count: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> str:
    """Launch a W&B sweep that reuses `train_ffnn` for each sampled config."""

    wandb_module = _ensure_wandb()
    if wandb_module is None:
        raise RuntimeError("wandb is required to launch sweeps. Install wandb and login before calling.")

    sweep_configs = build_wandb_sweep_configs()
    if method not in sweep_configs:
        available = ", ".join(sorted(sweep_configs))
        raise ValueError(f"Unknown sweep method '{method}'. Available methods: {available}.")

    sweep_config = copy.deepcopy(sweep_configs[method])
    if sweep_name:
        sweep_config.setdefault("name", sweep_name)
    if config_overrides:
        sweep_config = _deep_update_dict(sweep_config, config_overrides)

    input_dim = _infer_input_dim(X_train)
    num_classes = _infer_num_classes(y_train, y_valid)

    init_args = dict(init_kwargs) if init_kwargs else {}
    init_args.setdefault("project", project)
    if entity is not None:
        init_args.setdefault("entity", entity)

    def _trainable() -> None:
        with wandb_module.init(**init_args) as run:
            config = run.config
            activation, weights_init = _resolve_activation_init_from_config(config)
            run.config.update({"activation": activation, "weights_init": weights_init}, allow_val_change=True)
            print(f"wandb:\tweights_init: {weights_init}")
            model = FFNN(
                input_dim=input_dim,
                num_classes=num_classes,
                num_hidden_layers=int(config.get("num_hidden_layers", 1)),
                n_hidden_units=int(config.get("n_hidden_units", 128)),
                activation=activation,
                loss=str(config.get("loss", "cross_entropy")).lower(),
                learning_rate=float(config.get("learning_rate", 1e-3)),
                l2_coeff=float(config.get("l2_coeff", 0.0)),
                weights_init=weights_init,
                optimizer=str(config.get("optimizer", "adam")).lower(),
            )
            num_epochs = int(config.get("num_epochs", 10))
            batch_size = int(config.get("batch_size", 64))
            history = train_ffnn(
                model,
                X_train,
                y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                num_epochs=num_epochs,
                batch_size=batch_size,
                wandb_run=run,
            )

            if log_artifacts and X_test is not None and y_test is not None:
                artifacts = prepare_training_artifacts(
                    model,
                    history,
                    X_test=X_test,
                    y_test=y_test,
                    class_names=class_names,
                    dataset_name=dataset_name,
                )
                log_wandb_artifacts(
                    run,
                    artifacts,
                    extra_summary={"summary/train_epochs": num_epochs},
                )

    sweep_id = wandb_module.sweep(sweep_config, project=project, entity=entity)
    wandb_module.agent(sweep_id, function=_trainable, count=count)
    return sweep_id


def summarize_activation_init_performance(
    project: str,
    *,
    entity: Optional[str] = None,
    max_runs: Optional[int] = None,
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """Summarise validation performance grouped by activation/initializer pairs.

    Pulls runs from the specified W&B project, aggregates the recorded
    ``summary/activation`` and ``summary/weights_init`` pairs, and reports the
    mean/max validation accuracy across runs for each pairing. Returns a list of
    summary dictionaries (one per pairing) sorted by mean best validation
    accuracy. Optionally prints a compact table for quick inspection.
    """

    wandb_module = _ensure_wandb()
    if wandb_module is None:
        raise RuntimeError("wandb is required to summarise activation/init performance.")

    api = wandb_module.Api()
    project_path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(project_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch runs from '{project_path}': {exc}") from exc

    def _mean_max(values: List[Optional[float]]) -> Tuple[float, float]:
        arr = np.asarray([v for v in values if v is not None], dtype=np.float32)
        if arr.size == 0:
            return float(np.nan), float(np.nan)
        return float(np.nanmean(arr)), float(np.nanmax(arr))

    aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    processed = 0
    for run in runs:
        if max_runs is not None and processed >= max_runs:
            break
        summary = getattr(run, "summary", {})
        config = getattr(run, "config", {})
        activation = str(summary.get("summary/activation") or config.get("activation") or "").strip().lower()
        weights_init = str(summary.get("summary/weights_init") or config.get("weights_init") or "").strip().lower()
        if not activation or not weights_init:
            continue
        key = (activation, weights_init)
        best_valid_acc = summary.get("summary/best_valid_accuracy")
        final_valid_acc = summary.get("summary/final_valid_accuracy")
        final_train_acc = summary.get("summary/final_train_accuracy")
        aggregates.setdefault(
            key,
            {
                "activation": activation,
                "weights_init": weights_init,
                "best_valid_acc_values": [],
                "final_valid_acc_values": [],
                "final_train_acc_values": [],
            },
        )
        group = aggregates[key]
        group.setdefault("run_ids", []).append(run.id)
        group["best_valid_acc_values"].append(best_valid_acc if best_valid_acc is not None else None)
        group["final_valid_acc_values"].append(final_valid_acc if final_valid_acc is not None else None)
        group["final_train_acc_values"].append(final_train_acc if final_train_acc is not None else None)
        processed += 1

    summaries: List[Dict[str, Any]] = []
    for group in aggregates.values():
        best_mean, best_max = _mean_max(group["best_valid_acc_values"])
        final_val_mean, final_val_max = _mean_max(group["final_valid_acc_values"])
        final_train_mean, final_train_max = _mean_max(group["final_train_acc_values"])
        summaries.append(
            {
                "activation": group["activation"],
                "weights_init": group["weights_init"],
                "run_count": len(group.get("run_ids", [])),
                "best_valid_acc_mean": best_mean,
                "best_valid_acc_max": best_max,
                "final_valid_acc_mean": final_val_mean,
                "final_valid_acc_max": final_val_max,
                "final_train_acc_mean": final_train_mean,
                "final_train_acc_max": final_train_max,
                "run_ids": group.get("run_ids", []),
            }
        )

    summaries.sort(
        key=lambda item: (
            -np.nan_to_num(item["best_valid_acc_mean"], nan=-np.inf),
            item["activation"],
            item["weights_init"],
        )
    )

    if print_table:
        if not summaries:
            print("No runs with activation and weights_init summaries were found.")
        else:
            header = (
                f"{'Activation':<10} {'Init':<8} {'Runs':>4} "
                f"{'Best μ':>8} {'Best ↑':>8} {'Val μ':>8} {'Val ↑':>8} {'Train μ':>8}"
            )
            print("Activation/Initializer performance for", project_path)
            print(header)
            print("-" * len(header))

            def _fmt(value: float) -> str:
                return f"{value:.4f}" if not np.isnan(value) else "   NA"

            for entry in summaries:
                print(
                    f"{entry['activation']:<10} {entry['weights_init']:<8} {entry['run_count']:>4} "
                    f"{_fmt(entry['best_valid_acc_mean']):>8} {_fmt(entry['best_valid_acc_max']):>8} "
                    f"{_fmt(entry['final_valid_acc_mean']):>8} {_fmt(entry['final_valid_acc_max']):>8} "
                    f"{_fmt(entry['final_train_acc_mean']):>8}"
                )

    return summaries


def build_activation_init_summary_dataframe(
    summaries: List[Dict[str, Any]],
) -> Any:
    """Return a pandas DataFrame with human-friendly column names for summaries.

    This is a thin wrapper so notebooks for different datasets can reuse the
    same table layout when comparing activation/initializer pairs.
    """

    # Local import to avoid making pandas a hard dependency of this module
    import pandas as pd  # type: ignore

    summary_df = pd.DataFrame(summaries)
    renamed_df = summary_df.rename(
        columns={
            "activation": "Activation",
            "weights_init": "Weights Init",
            "run_count": "Run Count",
            "best_valid_acc_mean": "Best Validation Mean",
            "best_valid_acc_max": "Best Validation Max",
            "final_valid_acc_mean": "Final Validation Mean",
            "final_valid_acc_max": "Final Validation Max",
            "final_train_acc_mean": "Final Training Mean",
            "final_train_acc_max": "Final Training Max",
        },
    )
    ordered_columns = [
        "Activation",
        "Weights Init",
        "Run Count",
        "Best Validation Mean",
        "Best Validation Max",
        "Final Validation Mean",
        "Final Validation Max",
        "Final Training Mean",
        "Final Training Max",
    ]
    # Only keep columns that actually exist to be robust to missing keys
    existing_cols = [c for c in ordered_columns if c in renamed_df.columns]
    return renamed_df[existing_cols]


def create_loss_figure(
    history: Dict[str, List[float]], *, subtitle: Optional[str] = None, dataset_name: Optional[str] = None
) -> Optional[Any]:
    """Build a loss curve figure from the training history."""
    train_loss = history.get("train_loss", [])
    valid_loss = history.get("valid_loss", [])
    if not train_loss:
        return None
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train loss")
    if valid_loss:
        ax.plot(valid_loss, label="valid loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    main_title = "Training and validation loss"
    if dataset_name:
        main_title = f"{dataset_name}: {main_title}"
    if subtitle:
        ax.set_title(f"{main_title} ({subtitle})")
    else:
        ax.set_title(main_title)
    ax.legend()
    fig.tight_layout()
    return fig


def create_accuracy_figure(
    history: Dict[str, List[float]], *, subtitle: Optional[str] = None, dataset_name: Optional[str] = None
) -> Optional[Any]:
    """Build an accuracy curve figure from the training history."""
    train_acc = history.get("train_acc", [])
    valid_acc = history.get("valid_acc", [])
    if not train_acc:
        return None
    fig, ax = plt.subplots()
    ax.plot(train_acc, label="train accuracy")
    if valid_acc:
        ax.plot(valid_acc, label="valid accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    main_title = "Training and validation accuracy"
    if dataset_name:
        main_title = f"{dataset_name}: {main_title}"
    if subtitle:
        ax.set_title(f"{main_title} ({subtitle})")
    else:
        ax.set_title(main_title)
    ax.legend()
    fig.tight_layout()
    return fig


def create_gradient_norm_figure(
    history: Dict[str, List[List[float]]], model: FFNN, *, subtitle: Optional[str] = None, dataset_name: Optional[str] = None
) -> Optional[Any]:
    """Build a gradient norm figure from the training history."""
    grad_norms = history.get("grad_norms", [])
    if not grad_norms:
        return None
    grad_array = np.asarray(grad_norms, dtype=np.float32)
    if grad_array.ndim != 2 or grad_array.size == 0:
        return None
    fig, ax = plt.subplots()
    num_layers = grad_array.shape[1]
    for layer_idx in range(num_layers):
        if layer_idx < model.num_hidden_layers:
            label = f"Hidden layer {layer_idx + 1}"
        else:
            label = "Output layer"
        ax.plot(grad_array[:, layer_idx], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm (L2)")
    main_title = "Gradient norms across training"
    if dataset_name:
        main_title = f"{dataset_name}: {main_title}"
    if subtitle:
        ax.set_title(f"{main_title} ({subtitle})")
    else:
        ax.set_title(main_title)
    ax.legend()
    fig.tight_layout()
    return fig


def create_param_hist_figure(
    history: Dict[str, List[List[dict]]], model: FFNN, *, subtitle: Optional[str] = None, dataset_name: Optional[str] = None
) -> Optional[Any]:
    """Build parameter histogram figures for the final epoch."""
    histograms = history.get("param_histograms", [])
    if not histograms:
        return None
    final_hist = histograms[-1]
    if not final_hist:
        return None
    n_layers = len(final_hist)
    cols = min(3, n_layers)
    rows = int(np.ceil(n_layers / cols)) if cols else 0
    if rows == 0 or cols == 0:
        return None
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for layer_idx, layer_hist in enumerate(final_hist):
        counts = layer_hist.get("counts")
        bin_edges = layer_hist.get("bin_edges")
        if counts is None or bin_edges is None:
            continue
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        ax = axes[layer_idx // cols, layer_idx % cols]
        ax.bar(bin_centers, counts, width=widths, align="center")
        if layer_idx < model.num_hidden_layers:
            title = f"Hidden layer {layer_idx + 1} (weights+biases)"
        else:
            title = "Output layer (weights+biases)"
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    total_slots = rows * cols
    for idx in range(len(final_hist), total_slots):
        fig.delaxes(axes[idx // cols, idx % cols])
    if subtitle or dataset_name:
        main_title = "Parameter histograms"
        if dataset_name:
            main_title = f"{dataset_name}: {main_title}"
        if subtitle:
            fig.suptitle(f"{main_title} ({subtitle})", y=1.02)
        else:
            fig.suptitle(main_title, y=1.02)
    fig.tight_layout()
    return fig


def create_weight_bias_hist_figure(
    model: FFNN, bins: int = 30, *, subtitle: Optional[str] = None, dataset_name: Optional[str] = None
) -> Optional[Any]:
    """Plot separate histograms for weights and biases per layer."""
    num_layers = len(model.W)
    if num_layers == 0:
        return None
    fig, axes = plt.subplots(num_layers, 2, figsize=(8, 3 * num_layers))
    axes = np.atleast_2d(axes)
    for layer_idx, (W, b) in enumerate(zip(model.W, model.b)):
        weight_ax = axes[layer_idx, 0]
        bias_ax = axes[layer_idx, 1]
        weight_ax.hist(W.ravel(), bins=bins, color="tab:blue", alpha=0.8)
        bias_ax.hist(b.ravel(), bins=max(5, bins // 2), color="tab:orange", alpha=0.8)
        if layer_idx < model.num_hidden_layers:
            layer_name = f"Hidden layer {layer_idx + 1}"
        else:
            layer_name = "Output layer"
        layer_title = layer_name
        if dataset_name:
            layer_title = f"{dataset_name}: {layer_title}"
        weight_ax.set_title(f"{layer_title} weights")
        bias_ax.set_title(f"{layer_title} biases")
        weight_ax.set_xlabel("Value")
        bias_ax.set_xlabel("Value")
        weight_ax.set_ylabel("Count")
        bias_ax.set_ylabel("Count")
    if subtitle or dataset_name:
        main_title = "Weights and biases"
        if dataset_name:
            main_title = f"{dataset_name}: {main_title}"
        if subtitle:
            fig.suptitle(f"{main_title} ({subtitle})", y=1.02)
        else:
            fig.suptitle(main_title, y=1.02)
    fig.tight_layout()
    return fig


def evaluate_model_on_test(
    model: FFNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate the trained model on the test set and build confusion matrices."""
    X_test_arr = np.asarray(X_test)
    if X_test_arr.ndim > 2:
        X_test_flat = X_test_arr.reshape(X_test_arr.shape[0], -1)
    else:
        X_test_flat = X_test_arr
    y_test_arr = np.asarray(y_test)
    y_test_proba = model.predict_proba(X_test_flat)
    y_pred_labels = np.argmax(y_test_proba, axis=1)
    test_acc = accuracy(y_test_proba, y_test_arr)

    inferred_classes = model.num_classes
    if y_test_arr.size:
        inferred_classes = max(inferred_classes, int(y_test_arr.max()) + 1)
    if y_pred_labels.size:
        inferred_classes = max(inferred_classes, int(y_pred_labels.max()) + 1)

    conf_mat = np.zeros((inferred_classes, inferred_classes), dtype=int)
    for yt, yp in zip(y_test_arr, y_pred_labels):
        conf_mat[int(yt), int(yp)] += 1

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_mat_norm = conf_mat / row_sums

    class_names_list: Optional[List[str]] = None
    if class_names is not None:
        class_names_list = [str(name) for name in np.asarray(class_names).tolist()]

    return {
        "test_accuracy": test_acc,
        "y_true": y_test_arr,
        "y_pred_labels": y_pred_labels,
        "y_test_proba": y_test_proba,
        "confusion_matrix": conf_mat,
        "confusion_matrix_normalized": conf_mat_norm,
        "class_names": class_names_list,
    }


def create_confusion_matrix_figure(
    conf_mat_norm: np.ndarray,
    *,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    subtitle: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Any:
    """Build a confusion matrix heatmap figure."""
    class_labels = class_names if class_names is not None else [str(i) for i in range(conf_mat_norm.shape[0])]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf_mat_norm, cmap="Reds", vmin=0.0, vmax=1.0)
    main_title = title
    if dataset_name:
        main_title = f"{dataset_name}: {main_title}"
    if subtitle:
        ax.set_title(f"{main_title} ({subtitle})")  # add subtitle for confusion matrix (e.g., activation_function + weights_init)
    else:
        ax.set_title(main_title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)
    for i in range(conf_mat_norm.shape[0]):
        for j in range(conf_mat_norm.shape[1]):
            value = conf_mat_norm[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)
    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    return fig


def prepare_training_artifacts(
    model: FFNN,
    history: Dict[str, Any],
    *,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct reusable evaluation artifacts (figures, metrics, confusion table)."""

    artifacts: Dict[str, Any] = {
        "figures": {},
        "metrics": {},
        "summary": {},
        "confusion_table": None,
    }

    subtitle = f"{model.weights_init} + {model.act.name}" #subtitle  (weights_init + activation)

    acc_fig = create_accuracy_figure(history, subtitle=subtitle, dataset_name=dataset_name)
    if acc_fig is not None:
        artifacts["figures"]["accuracy_curves"] = acc_fig

    loss_fig = create_loss_figure(history, subtitle=subtitle, dataset_name=dataset_name)
    if loss_fig is not None:
        artifacts["figures"]["loss_curves"] = loss_fig

    grad_fig = create_gradient_norm_figure(history, model, subtitle=subtitle, dataset_name=dataset_name)
    if grad_fig is not None:
        artifacts["figures"]["gradient_norms"] = grad_fig

    hist_fig = create_param_hist_figure(history, model, subtitle=subtitle, dataset_name=dataset_name)
    if hist_fig is not None:
        artifacts["figures"]["param_histograms"] = hist_fig

    weight_bias_fig = create_weight_bias_hist_figure(model, subtitle=subtitle, dataset_name=dataset_name)
    if weight_bias_fig is not None:
        artifacts["figures"]["weight_bias_histograms"] = weight_bias_fig

    eval_payload = evaluate_model_on_test(model, X_test, y_test, class_names=class_names)
    test_acc = eval_payload["test_accuracy"]
    artifacts["metrics"]["metrics/test_accuracy"] = test_acc
    artifacts["summary"]["summary/test_accuracy"] = test_acc
    artifacts["confusion_matrix"] = eval_payload["confusion_matrix"]
    artifacts["confusion_matrix_normalized"] = eval_payload["confusion_matrix_normalized"]
    artifacts["predictions"] = {
        "y_true": eval_payload["y_true"],
        "y_pred_labels": eval_payload["y_pred_labels"],
    }

    class_labels = eval_payload["class_names"]
    if class_labels is None:
        class_labels = [str(i) for i in range(eval_payload["confusion_matrix"].shape[0])]
    artifacts["class_names"] = class_labels
    conf_fig = create_confusion_matrix_figure(
        eval_payload["confusion_matrix_normalized"],
        class_names=class_labels,
        title="Confusion Matrix",
        subtitle=subtitle,
        dataset_name=dataset_name,
    )
    artifacts["figures"]["confusion_matrix"] = conf_fig

    artifacts["confusion_table"] = {
        "probs": None,
        "y_true": eval_payload["y_true"].tolist(),
        "preds": eval_payload["y_pred_labels"].tolist(),
        "class_names": class_labels,
    }

    # Activation/initializer pairing summary for quick comparison across runs.
    artifacts["summary"]["summary/activation"] = model.act.name
    artifacts["summary"]["summary/weights_init"] = model.weights_init

    return artifacts


def log_wandb_artifacts(
    wandb_run: Optional[Any],
    artifacts: Dict[str, Any],
    *,
    extra_summary: Optional[Dict[str, Any]] = None,
    extra_logs: Optional[Dict[str, Any]] = None,
) -> None:
    """Log prepared artifacts to a wandb run if available."""

    if wandb_run is None:
        return
    wandb_module = _ensure_wandb()
    if wandb_module is None:
        return

    log_payload: Dict[str, Any] = {}
    for key, value in (artifacts.get("metrics") or {}).items():
        log_payload[key] = value
    for key, fig in (artifacts.get("figures") or {}).items():
        log_payload[key] = wandb_module.Image(fig)
    if extra_logs:
        log_payload.update(extra_logs)
    if log_payload:
        wandb_run.log(log_payload)

    confusion_payload = artifacts.get("confusion_table")
    if confusion_payload and hasattr(wandb_module, "plot") and hasattr(wandb_module.plot, "confusion_matrix"):
        wandb_run.log(
            {
                "plots/confusion_matrix_table": wandb_module.plot.confusion_matrix(**confusion_payload)
            }
        )

    summary_updates = dict(artifacts.get("summary") or {})
    if extra_summary:
        summary_updates.update(extra_summary)
    if summary_updates:
        wandb_run.summary.update(summary_updates)
