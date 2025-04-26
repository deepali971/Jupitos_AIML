import torch
import torch.nn.functional as F

# Self-Attention
# This module implements a self-attention mechanism.

def self_attention(x):
    """
    Computes self-attention for the input tensor.
    
    Args:
        x (tensor): Input tensor of shape (sequence_length, batch_size, embed_size).
        
    Returns:
        tensor: The output of the self-attention mechanism.
    """
    Q = K = V = x  # In self-attention, Q, K, and V are the same
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1]))
    attn_weights = F.softmax(scores, dim=-1)  # Apply softmax to get attention weights
    return torch.matmul(attn_weights, V)  # Multiply with Value matrix

# Example usage
if __name__ == "__main__":
    embed_size = 768
    input_tensor = torch.randn(10, 1, embed_size)  # Sequence length of 10, batch size of 1
    
    attn_output = self_attention(input_tensor)
    print("Self-Attention Output Shape:", attn_output.shape)
