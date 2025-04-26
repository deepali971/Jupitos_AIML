import torch
import torch.nn as nn

# Decoder
# This module implements a simple decoder for the Transformer model.

class Decoder(nn.Module):
    """Decoder block for the Transformer model."""
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.encoder_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.add_norm = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        attention_output, _ = self.attention(x, x, x)
        x = self.add_norm(x + attention_output)  # Add & Norm
        enc_attention_output, _ = self.encoder_attention(x, enc_out, enc_out)
        x = self.add_norm(x + enc_attention_output)  # Add & Norm
        forward_output = self.feed_forward(x)
        return self.add_norm(x + self.dropout(forward_output))  # Add & Norm

# Example usage
if __name__ == "__main__":
    embed_size = 768
    heads = 8
    decoder = Decoder(embed_size, heads)
    
    # Simulated input embeddings and encoder output
    input_embeddings = torch.randn(10, 1, embed_size)  # Sequence length of 10, batch size of 1
    encoder_output = torch.randn(10, 1, embed_size)  # Simulated encoder output
    decoder_output = decoder(input_embeddings, encoder_output)
    print("Decoder Output Shape:", decoder_output.shape)
