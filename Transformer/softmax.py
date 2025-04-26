import torch
import torch.nn.functional as F

# Softmax
# This module implements the softmax function to convert logits into probabilities.

def softmax(logits):
    """
    Applies the softmax function to the input logits.
    
    Args:
        logits (tensor): Raw scores from the model.
        
    Returns:
        tensor: Probabilities corresponding to the logits.
    """
    return F.softmax(logits, dim=-1)  # Converts logits into probabilities

# Example usage
if __name__ == "__main__":
    logits = torch.randn(1, 10)  # Simulated logits
    probs = softmax(logits)
    print("Softmax Probabilities:", probs)
