import torch
import torch.nn as nn

# Encoder
# This module implements a simple encoder for the Transformer model.

class Encoder(nn.Module):
    """Encoder block for the Transformer model."""
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.add_norm = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.add_norm(x + attention_output)  # Add & Norm
        forward_output = self.feed_forward(x)
        return self.add_norm(x + self.dropout(forward_output))  # Add & Norm

# Example usage
if __name__ == "__main__":
    embed_size = 768
    heads = 8
    encoder = Encoder(embed_size, heads)
    
    # Simulated input embeddings
    input_embeddings = torch.randn(10, 1, embed_size)  # Sequence length of 10, batch size of 1
    encoder_output = encoder(input_embeddings)
    print("Encoder Output Shape:", encoder_output.shape)
