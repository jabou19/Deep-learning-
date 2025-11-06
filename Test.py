import numpy as np
from ffnn import FeedForwardNN

# Create random input data (batch_size=5, input_dim=4)
X = np.random.randn(5, 4)

# Define network structure: input(4) -> hidden(5) -> output(3)
ffnn = FeedForwardNN(layer_sizes=[4, 5, 3], activation='relu', output_activation='softmax')

# Run forward pass
y_pred, cache = ffnn.forward(X)

print("✅ Forward pass successful!")
print("Input shape:", X.shape)
print("Output shape:", y_pred.shape)
print("Predictions (probabilities):")
print(y_pred)
print("Sum per sample (should ≈ 1):", np.sum(y_pred, axis=1))
