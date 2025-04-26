#4. Self-Attention Encoding
import torch
import torch.nn.functional as F
import math # Import the math module

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Sample input tensors
Q = torch.rand(2, 4, 4)  # Batch size 2, sequence length 4, dimension 4
K = torch.rand(2, 4, 4)
V = torch.rand(2, 4, 4)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Self-Attention Output:")
print(output)
print("Attention Weights:")
print(attention_weights)
