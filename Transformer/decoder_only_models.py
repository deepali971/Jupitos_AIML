import torch
import torch.nn as nn

# Decoder-Only Models
# This module implements a simple decoder-only architecture.

class DecoderOnlyModel(nn.Module):
    """Simple Decoder-Only model."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, dec_input):
        return self.decoder(dec_input, dec_input)  # Using decoder input as both input and output

# Example usage
if __name__ == "__main__":
    embed_size = 768
    heads = 8
    decoder = Decoder(embed_size, heads)
    model = DecoderOnlyModel(decoder)
    
    # Simulated input for decoder
    dec_input = torch.randn(10, 1, embed_size)  # Decoder input
    
    output = model(dec_input)
    print("Decoder-Only Model Output Shape:", output.shape)
