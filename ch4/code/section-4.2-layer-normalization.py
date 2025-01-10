import torch
import torch.nn as nn
import tiktoken

torch.manual_seed(123)

# Create 2 training samples with 5 dimensions/features each
batch_example = torch.randn(2, 5)
print(f'batch_example:\n{batch_example}')

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(f'output:\n{out}')

'''
batch_example:
tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
        [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]])
output:
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
'''

# Compute its means and variance
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')

'''
mean: tensor([[0.1324],
        [0.2170]], grad_fn=<MeanBackward1>)
var: tensor([[0.0231],
        [0.0398]], grad_fn=<VarBackward0>)
'''

# Applying layer norm
out_norm = (out - mean) / torch.sqrt(var)
print(f'out_norm:\n{out_norm}')
'''
out_norm:
tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
       grad_fn=<DivBackward0>)
'''

# Recompute mean and variance again
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')

'''
mean: tensor([[-5.9605e-08],
        [ 1.9868e-08]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
'''

torch.set_printoptions(sci_mode=False)
print(f'mean: {mean}')
print(f'var: {var}')

'''
mean: tensor([[    -0.0000],
        [     0.0000]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
'''