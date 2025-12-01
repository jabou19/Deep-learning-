## Implementing a Neural Network from Scratch with NumPy

**Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)**

### Overview

This project implements a **fully-connected feedforward neural network (FFNN)** from scratch using **NumPy** only. No deep learning frameworks such as TensorFlow or PyTorch are used. All core components are implemented manually:

- Forward and backward propagation
- Gradient-based optimisation (mini-batch gradient descent / Adam)
- Loss computation with optional L2 regularisation
- Evaluation and experiment tracking with **Weights & Biases (WandB)**

The overall goal is to understand how modern deep learning libraries work internally by building and experimenting with a NumPy-only implementation.

### Objectives

- Design and train a configurable FFNN for classification using NumPy.
- Support image-based datasets such as **Fashion-MNIST** and **CIFAR-10**.
- Implement from scratch:
	- Forward pass (matrix multiplications + activation functions)
	- Loss computation (cross-entropy) with L2 regularisation
	- Backward pass (manual derivatives and gradient computation)
	- Training loop with mini-batch gradient descent / Adam
- Evaluate models using accuracy, loss curves, and confusion matrices.
- Track experiments with **WandB**, including hyperparameter sweeps.

### Datasets

The main experiments are run on two open-source datasets:

- **Fashion-MNIST** – grayscale clothing images (28×28, 10 classes).
- **CIFAR-10** – colour images (32×32×3, 10 classes).

Both datasets are small enough for CPU-based NumPy training but rich enough to demonstrate overfitting, regularisation effects, and the impact of different optimisers and inits. Additional open-source datasets (e.g. from Kaggle) can be plugged in if desired.

### FFN Architecture and Hyperparameters

The central component is a flexible `FFNN` class with the following configurable hyperparameters:

- `num_epochs`
- `num_hidden_layers`
- `n_hidden_units`
- `learning_rate`
- `optimizer` (Adam)
- `batch_size`
- `l2_coeff`
- `weights_init` ( Xavier, He)
- `activation` ( ReLU, tanh, sigmoid)


### Implementation Stages

1. **Forward pass** – linear layers + activations using NumPy matrix operations.
2. **Loss computation** –  cross-entropy with optional L2 penalty on weights.
3. **Backward pass** – manual derivatives for all layers and activations.
4. **Parameter update** – gradient descent / Adam using accumulated gradients.
5. **Training loop** – mini-batch iteration over the dataset with periodic validation.
6. **Evaluation** – compute accuracy and loss curves on train/val/, and a confusion matrix on the test set.

### Experiment Logging with WandB

Each training run can be logged to **Weights & Biases**. Logged artefacts include:

- Learning curves: `train_loss`, `val_loss`, `train_acc`, `val_acc`.
- Parameter histograms and gradient norms over time.
- Hyperparameter sweeps (Bayesian) over:
	- network depth and width
	- activation functions
	- weight initialisation
	- optimisers and learning rates
- Summary tables and bar plots comparing activations and initialisations across runs.

### Project Structure

Repository root (this folder):

```
Deep-learning-/
│
├── CIFAR-10.ipynb              # CIFAR-10 loading, training, sweeps, summaries
├── Fashion-MNIST .ipynb        # Fashion-MNIST loading, training, sweeps, summaries
├── Utilisfunction.py           # FFNN implementation, training helpers, WandB utilities
├── data/                       # Fashion-MNIST raw gzip files (downloaded)
├── Dataset/                    # CIFAR-10 python batches (pre-downloaded)
├── wandb/                      # Local WandB run logs (auto-created)
└── README.md                   # Project documentation
```

### How to Run

#### 1. Install dependencies

Create and activate a Python environment, then install the required packages:

```powershell
pip install numpy matplotlib pandas wandb
```

#### 2. Run the Fashion-MNIST.ipynb

1. Open `Fashion-MNIST.ipynb` in Jupyter or VS Code.
2. Run the setup / data-loading cells.
3. Run the training / sweep cells to launch FFNN on Fashion-MNIST.
4. Run the final summary cell to compare activation functions and initialisations (local-only analysis).

#### 3. Run the CIFAR-10.ipynb

1. Open `CIFAR-10.ipynb`.
2. Run the setup / data-loading cells (these use the files in `Dataset/`).
3. Run the training / sweep cells to launch FFNN on CIFAR-10.
4. Run the final summary cell to compare activation functions and initialisations (local-only analysis).

Fashion-MNIST and CIFAR-10 are run **separately** in their own notebooks. Both use the shared implementation in `Utilisfunction.py`.

#### 4. Enable WandB logging 
Before running sweeps, log in to WandB in a terminal:

```powershell
wandb login
```

Then, when you run the sweep cells in each notebook, runs will be tracked in your WandB account (metrics, curves, histograms, confusion matrices, and hyperparameter sweeps).
