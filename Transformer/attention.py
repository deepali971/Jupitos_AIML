import torch
import torch.nn.functional as F

# Attention
# This module implements a simple attention mechanism.

def attention(Q, K, V):
    """
    Computes scaled dot-product attention.
    
    Args:
        Q (tensor): Query matrix.
        K (tensor): Key matrix.
        V (tensor): Value matrix.
        
    Returns:
        tensor: The output of the attention mechanism.
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1]))
    attn_weights = F.softmax(scores, dim=-1)  # Apply softmax to get attention weights
    return torch.matmul(attn_weights, V)  # Multiply with Value matrix

# Example usage
if __name__ == "__main__":
    embed_size = 768
    Q = torch.randn(1, 10, embed_size)  # Query
    K = torch.randn(1, 10, embed_size)  # Key
    V = torch.randn(1, 10, embed_size)  # Value
    
    attn_output = attention(Q, K, V)
    print("Attention Output Shape:", attn_output.shape)
