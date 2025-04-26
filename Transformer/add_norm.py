import torch
import torch.nn as nn

# Add & Norm
# This module implements a layer that adds a residual connection and normalizes the output.

class AddNorm(nn.Module):
    """Layer that adds residual connections followed by layer normalization."""
    def __init__(self, size):
        super().__init__()
        self.norm = nn.LayerNorm(size)
    
    def forward(self, x, sublayer):
        return self.norm(x + sublayer)  # Residual connection followed by normalization

# Example usage
if __name__ == "__main__":
    embed_size = 768
    add_norm_layer = AddNorm(embed_size)
    
    # Simulated embeddings and attention output
    embeddings = torch.randn(1, 10, embed_size)  # Batch size of 1, sequence length of 10
    attn_output = torch.randn(1, 10, embed_size)
    
    norm_output = add_norm_layer(embeddings, attn_output)
    print("Add & Norm Output Shape:", norm_output.shape)
