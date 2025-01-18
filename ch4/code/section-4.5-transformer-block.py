import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from config import GPT_CONFIG_124M

torch.manual_seed(123)

# Create sample input of shape: [batch_size, num_tokens, emb_dim]
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape) # torch.Size([2, 4, 768])
print("Output shape:", output.shape) # torch.Size([2, 4, 768])
