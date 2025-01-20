import torch
import torch.nn as nn
from gpt_model import GPTModel 

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

torch.manual_seed(123)

# Input is tokenised text
batch = torch.tensor([
        [6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print('Input batch:', batch.shape, '\n')
print('Input batch:\n', batch)

print('\nOutput shape:', out.shape)
print(out)

'''
Input batch: torch.Size([2, 4]) 

Input batch:
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

Output shape: torch.Size([2, 4, 50257])
tensor([[[ 0.4398, -1.1968, -0.3533,  ..., -0.1638, -1.2250,  0.0803],
         [ 0.1247, -2.2218, -0.6962,  ..., -0.5499, -1.4728,  0.0665],
         [ 0.5515, -1.5762, -0.3643,  ...,  0.0276, -1.7843, -0.2937],
         [-0.8036, -1.6966, -0.2890,  ...,  0.3314, -1.2682,  0.1784]],

        [[-0.3290, -1.8522, -0.1652,  ..., -0.1751, -1.0380, -0.2999],
         [-0.0083, -1.2779, -0.1241,  ...,  0.3117, -1.4347,  0.2552],
         [ 0.5651, -1.1005, -0.1858,  ...,  0.1592, -1.2875,  0.2329],
         [-0.5593, -1.3399,  0.3970,  ...,  0.8095, -1.6276,  0.3201]]],
       grad_fn=<UnsafeViewBackward0>)
'''

# Computing the number of parameters in the model's parameter tensors:
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}") 
# Total number of parameters: 163,009,536

# Shape of the token embedding layers and linear output layers
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

'''
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])
'''

# Remove the reused weight
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}")
'''
Number of trainable parameters considering weight tying: 124,412,160
'''

# Compute the memory requirements of the 163 million parameters 
total_size_bytes = total_params * 4 # assuming float32 i.e. 4 bytes
total_size_mb = total_size_bytes / (1024*1024) # convert to MB
print(f"Total size of the model: {total_size_bytes:,} B or {total_size_mb:.2f} MB")
