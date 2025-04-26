import torch
import torch.nn as nn

# Encoder-Decoder Models
# This module implements a simple encoder-decoder architecture.

class EncoderDecoderModel(nn.Module):
    """Simple Encoder-Decoder model."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, dec_input):
        enc_output = self.encoder(enc_input)
        return self.decoder(dec_input, enc_output)

# Example usage
if __name__ == "__main__":
    embed_size = 768
    heads = 8
    encoder = Encoder(embed_size, heads)
    decoder = Decoder(embed_size, heads)
    model = EncoderDecoderModel(encoder, decoder)
    
    # Simulated input for encoder and decoder
    enc_input = torch.randn(10, 1, embed_size)  # Encoder input
    dec_input = torch.randn(10, 1, embed_size)  # Decoder input
    
    output = model(enc_input, dec_input)
    print("Encoder-Decoder Model Output Shape:", output.shape)
