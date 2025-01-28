import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.0455715 * torch.pow(x,3)))
        )
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), # 768 x 3072
            GELU(),
            nn.Linear(4 * cfg['emb_din'], cfg['emb_dim']) # 3072 x 768
        )

    def forward(self, x):
        return self.layers(x)
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 768
        self.shift = nn.Parameter(torch.ones(emb_dim)) # 768

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False -> in the variance calculation, n is used instead of n-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
