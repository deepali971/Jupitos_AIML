import torch
import torch.nn.functional as F

# Masked Attention
# This module implements masked attention to prevent future tokens from influencing past ones.

def masked_attention(Q, K, V, mask):
    """
    Applies masking to prevent future tokens from being attended to.
    
    Args:
        Q (tensor): Query matrix.
        K (tensor): Key matrix.
        V (tensor): Value matrix.
        mask (tensor): Mask to apply.
        
    Returns:
        tensor: The output of the masked attention mechanism.
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1]))
    scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply mask
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)

# Example usage
if __name__ == "__main__":
    embed_size = 768
    Q = torch.randn(1, 10, embed_size)  # Query
    K = torch.randn(1, 10, embed_size)  # Key
    V = torch.randn(1, 10, embed_size)  # Value
    mask = torch.ones(Q.shape[:-1])  # Create a mask
    
    masked_output = masked_attention(Q, K, V, mask)
    print("Masked Attention Output Shape:", masked_output.shape)
