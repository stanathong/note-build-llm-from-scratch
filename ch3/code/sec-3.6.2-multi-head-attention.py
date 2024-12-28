import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
               'd_out must be divisible by num_heads'
        
        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce the projection dim to match the desired output dim
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  
        # Use the linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out) 
        self.droput = nn.Dropout(dropout)
        self.register_buffer(
            # This is dictionary
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)      # (b, num_tokens, d_out) 
        queries = self.W_query(x) # (b, num_tokens, d_out) 
        values = self.W_value(x)  # (b, num_tokens, d_out) 

        #  Transform (b, num_tokens, d_out) to 
        #            (b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Tranform (b, num_tokens, self.num_heads, self.head_dim) to 
        #          (b, self.num_heads, num_tokens, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute the dot product for each head
        # queries: (b, self.num_heads, num_tokens, self.head_dim) @
        # keys:    (b, self.num_heads, self.head_dim, num_tokens)
        #       =  (b, self.num_heads, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(2, 3)
        
        # Mask truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.droput(attention_weights)

        # Compute context vector based on atteiont weights and values
        # attention_weights @ values =
        #    (b, self.num_heads, num_tokens, num_tokens) @
        #        (b, self.num_heads, num_tokens, self.head_dim)
        #        0       1             2            3
        #    =  (b, self.num_heads, num_tokens, self.head_dim)
        # transpose(1, 2) -> (b, num_tokens, self.num_heads, self.head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)
        # Combine heads where self.d_out = self.num_heads * self.head_dim
        # Note: .contiguous() ensure tensor’s data is reorganized in memory 
        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.d_out)

        # Add optional linear projection
        context_vector = self.out_proj(context_vector)
        return context_vector

# TO BE ABLE TO COMPARE WITH SEC 3.6.1, discard out_proj layer
# It has shown that the results are different.
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
num_heads = 2
d_out_per_head = 2
d_out = num_heads * d_out_per_head # The output embedding size, d_out = 2

# 2-head attention
multi_head_attention = MultiHeadAttention(
    d_in, d_out, context_length, dropout=0.0, num_heads=2)

context_vectors = multi_head_attention(batch)
print('context_vectors.shape :', context_vectors.shape)
print(context_vectors)

# Not the same resul as 3.6.1
'''
context_vectors.shape : torch.Size([2, 6, 4])
tensor([[[-0.3132, -0.2272,  0.4772,  0.1063],
         [-0.2308,  0.0329,  0.5764,  0.3007],
         [-0.2059,  0.1190,  0.6097,  0.3654],
         [-0.1642,  0.1340,  0.5431,  0.3503],
         [-0.1689,  0.1794,  0.5296,  0.3389],
         [-0.1407,  0.1699,  0.5040,  0.3403]],

        [[-0.3132, -0.2272,  0.4772,  0.1063],
         [-0.2308,  0.0329,  0.5764,  0.3007],
         [-0.2059,  0.1190,  0.6097,  0.3654],
         [-0.1642,  0.1340,  0.5431,  0.3503],
         [-0.1689,  0.1794,  0.5296,  0.3389],
         [-0.1407,  0.1699,  0.5040,  0.3403]]], grad_fn=<ViewBackward0>)
'''

# Result with out_proj layer
'''
context_vectors.shape : torch.Size([2, 6, 4])
tensor([[[ 0.1184,  0.3120, -0.0847, -0.5774],
         [ 0.0178,  0.3221, -0.0763, -0.4225],
         [-0.0147,  0.3259, -0.0734, -0.3721],
         [-0.0116,  0.3138, -0.0708, -0.3624],
         [-0.0117,  0.2973, -0.0698, -0.3543],
         [-0.0132,  0.2990, -0.0689, -0.3490]],

        [[ 0.1184,  0.3120, -0.0847, -0.5774],
         [ 0.0178,  0.3221, -0.0763, -0.4225],
         [-0.0147,  0.3259, -0.0734, -0.3721],
         [-0.0116,  0.3138, -0.0708, -0.3624],
         [-0.0117,  0.2973, -0.0698, -0.3543],
         [-0.0132,  0.2990, -0.0689, -0.3490]]], grad_fn=<ViewBackward0>)
'''