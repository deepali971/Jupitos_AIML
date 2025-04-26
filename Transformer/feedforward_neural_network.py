import torch
import torch.nn as nn

# Feedforward Neural Network (FNN)
# This module implements a simple feedforward neural network used in Transformer models.

class FeedForwardNN(nn.Module):
    """Feedforward layer used in Transformer models."""
    def __init__(self, embed_size):
        super().__init__()
        self.fc = nn.Linear(embed_size, embed_size)  # Fully connected layer
    
    def forward(self, x):
        return self.fc(F.relu(x))  # Applies ReLU activation before passing through linear layer

# Example usage
if __name__ == "__main__":
    embed_size = 768
    ffn = FeedForwardNN(embed_size)
    
    # Simulated input
    input_tensor = torch.randn(1, embed_size)  # Batch size of 1
    ffn_output = ffn(input_tensor)
    print("Feedforward Neural Network Output Shape:", ffn_output.shape)
