import torch
import torch.nn.functional as F

# Output Probabilities / Logits
# This module converts logits into probabilities.

def output_probabilities(logits):
    """
    Converts raw scores (logits) into probabilities using softmax.
    
    Args:
        logits (tensor): Raw scores from the model.
        
    Returns:
        tensor: Probabilities corresponding to the logits.
    """
    return F.softmax(logits, dim=-1)  # Converts logits into probabilities

# Example usage
if __name__ == "__main__":
    logits = torch.randn(1, 10)  # Simulated logits
    probs = output_probabilities(logits)
    print("Output Probabilities:", probs)
