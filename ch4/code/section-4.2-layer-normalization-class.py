import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # In the variance calculation, we devide by n instead of n-1.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

torch.manual_seed(123)

# Create 2 training samples with 5 dimensions/features each
batch_example = torch.randn(2, 5)
print(f'batch_example:\n{batch_example}')

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
# Compute mean and variance
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')

'''
mean: tensor([[1.0000],
        [1.0000]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
'''