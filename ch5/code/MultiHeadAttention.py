# https://github.com/stanathong/note-build-llm-from-scratch/tree/main/ch3

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, droptout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
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
        self.dropout = nn.Dropout(droptout)
        self.register_buffer(
            # This is dictionary
            'mask',
            # num_tokens x num_token
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values = self.W_value(x) # (b, num_tokens, d_out)

        #  Transform (b, num_tokens, d_out) to 
        #            (b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #           0      1            2               3
        # Tranform (b, num_tokens, self.num_heads, self.head_dim) to 
        #          (b, self.num_heads, num_tokens, self.head_dim)
        keys = keys.transpose(1, 2)        
        queries = queries.transpose(1, 2)  
        values = values.transpose(1, 2)  

        # Compute the dot product for each head
        # queries: (b, self.num_heads, num_tokens, self.head_dim) @
        # keys.T:  (b, self.num_heads, self.head_dim, num_tokens)
        # quereis @ keys.T(2,3) = (b, self.num_heads, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(2, 3)

        # Mask truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute attention weights
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] **0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute context vector based on attention weights and values
        # attention_weights @ values =
        #        (b, self.num_heads, num_tokens, num_tokens) @
        #        (b, self.num_heads, num_tokens, self.head_dim)
        #        0       1             2            3
        #    =  (b, self.num_heads, num_tokens, self.head_dim)
        # transpose(1, 2) -> (b, num_tokens, self.num_heads, self.head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)    

        # Combine heads where self.d_out = self.num_heads * self.head_dim
        # Note: .contiguous() ensure tensor’s data is re-organised in memory 
        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.d_out)
        
        # Add optional linear projection
        context_vector = self.out_proj(context_vector)
        return context_vector
