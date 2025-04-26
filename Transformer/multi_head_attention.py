import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Attention
# This module implements a multi-head self-attention mechanism.

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.W_q = nn.Linear(embed_size, embed_size)  # Linear transformation for Query
        self.W_k = nn.Linear(embed_size, embed_size)  # Linear transformation for Key
        self.W_v = nn.Linear(embed_size, embed_size)  # Linear transformation for Value
        self.fc_out = nn.Linear(embed_size, embed_size)  # Final linear layer

    def forward(self, x):
        batch_size = x.shape[0]
        Q = self.W_q(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        attn_weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(attn_weights, V)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        return self.fc_out(attention)

# Example usage
if __name__ == "__main__":
    embed_size = 768
    heads = 8
    mha = MultiHeadAttention(embed_size, heads)
    
    input_tensor = torch.randn(1, 10, embed_size)  # Batch size of 1, sequence length of 10
    mha_output = mha(input_tensor)
    print("Multi-Head Attention Output Shape:", mha_output.shape)
