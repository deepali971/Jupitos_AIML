import torch
import torch.nn as nn

# Encoder-Decoder Attention
# This module implements attention mechanism between encoder and decoder.

class EncoderDecoderAttention(nn.Module):
    """Attention mechanism for Encoder-Decoder models."""
    def __init__(self, embed_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=8)
    
    def forward(self, enc_out, dec_in):
        return self.attention(dec_in, enc_out, enc_out)[0]  # Return only the output

# Example usage
if __name__ == "__main__":
    embed_size = 768
    encoder_decoder_attn = EncoderDecoderAttention(embed_size)
    
    # Simulated encoder output and decoder input
    enc_out = torch.randn(10, 1, embed_size)  # Encoder output
    dec_in = torch.randn(10, 1, embed_size)  # Decoder input
    
    ed_attn_output = encoder_decoder_attn(enc_out, dec_in)
    print("Encoder-Decoder Attention Output Shape:", ed_attn_output.shape)
