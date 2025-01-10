import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))) 
        )

# Implementation of feed forward neural network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), # 768 x 3072
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']), # 3072 x 768
        )
    
    def forward(self, x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M)
# Create a sample input of batch size = 2, 3 tokens, token embedding size = 768
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape) # torch.Size([2, 3, 768])
