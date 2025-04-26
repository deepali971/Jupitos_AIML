#6. Residual & Layer Encoding
#Residual Connection
import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.norm(x + self.layer(x))

# Sample input tensor
input_tensor = torch.rand(5, 10)  # Batch size 5, input dimension 10

# Create Residual Layer
residual_layer = ResidualLayer(10)

# Forward pass
output = residual_layer(input_tensor)

print("Residual Layer Output:")
print(output)
