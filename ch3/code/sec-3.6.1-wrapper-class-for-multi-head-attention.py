import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Add dropout
        self.dropout = nn.Dropout(dropout)
        # reigister_buffers helps with moving buffers to the appropriate devices (CPU/GPU)
        self.register_buffer(
            'mask', # name
            # buffer to register
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs):
        b, num_tokens, d_in = inputs.shape
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        # Since index 0 is batch, we will do transpose between dim 1 and 2
        # keeping the batch dimension at the first position (0)
        attention_scores = queries @ keys.transpose(1,2)
        # In Pytorch, operation with _ trailing are performed in-place
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Contain a list of self-attention
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)]
        )
    
    def forward(self, x):
        # Concat the results along the column axis
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
print('batch.shape:', batch.shape) # torch.Size([2, 6, 3])

context_length = batch.shape[1] # 6
d_in = inputs.shape[-1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

# 2-head attention
multi_head_attention = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, dropout=0.0, num_heads=2)

context_vectors = multi_head_attention(batch)
print('context_vectors.shape :', context_vectors.shape)
print(context_vectors)

'''
context_vectors.shape : torch.Size([2, 6, 4])
tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
'''
