import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @self.W_value

        attention_scores = queries @ keys.T # omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vector = attention_weights @ values
        return context_vector


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # changed from v1
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x) # changed from v1 which do `x @ self.W_key`
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T # omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vector = attention_weights @ values
        return context_vector
    

# Initialise input embedding sequence
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
d_in = inputs.shape[1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

# Create an instance for both class
torch.manual_seed(123)
self_attention_v1 = SelfAttention_v1(d_in, d_out)

torch.manual_seed(123)
self_attention_v2 = SelfAttention_v2(d_in, d_out)

# nn.Linear stores the weight matrix in a transposed form, hence transpose is required here.
self_attention_v1.W_query = torch.nn.Parameter(self_attention_v2.W_query.weight.T)
self_attention_v1.W_query = torch.nn.Parameter(self_attention_v2.W_query.weight.T)
self_attention_v1.W_value = torch.nn.Parameter(self_attention_v2.W_value.weight.T)

print('Output from v1:\n', self_attention_v1(inputs))
print('Output from v2:\n', self_attention_v2(inputs))

'''
Output from v1:
 tensor([[-0.5323, -0.1086],
        [-0.5253, -0.1062],
        [-0.5254, -0.1062],
        [-0.5253, -0.1057],
        [-0.5280, -0.1068],
        [-0.5243, -0.1055]], grad_fn=<MmBackward0>)
Output from v2:
 tensor([[-0.5337, -0.1051],
        [-0.5323, -0.1080],
        [-0.5323, -0.1079],
        [-0.5297, -0.1076],
        [-0.5311, -0.1066],
        [-0.5299, -0.1081]], grad_fn=<MmBackward0>)
'''