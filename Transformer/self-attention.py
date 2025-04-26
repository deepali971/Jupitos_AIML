#4.Self-Attention
import torch
import torch.nn.functional as F

# Input: sequence of word embeddings (batch_size, seq_len, d_model)
X = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])  # Example input

# Linear transformations for Q, K, V
d_k = X.size(-1)
W_Q = torch.randn(d_k, d_k)
W_K = torch.randn(d_k, d_k)
W_V = torch.randn(d_k, d_k)

Q = torch.matmul(X, W_Q)
K = torch.matmul(X, W_K)
V = torch.matmul(X, W_V)

# Scaled dot-product attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
attention_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attention_weights, V)

print("Output:", output)