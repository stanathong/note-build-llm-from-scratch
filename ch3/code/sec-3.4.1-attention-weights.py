import torch

# Create a sequence of token embeddings with 3 dimension
# Input sentence: Your journey starts with one step
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)    
)

print('inputs.shape:', inputs.shape) # torch.Size([6, 3])

x_2 = inputs[1] # journey
d_in = inputs.shape[1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

# Initialise the three weight matrices: Wq, Wk, Wv.
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print(W_query.shape) # torch.Size([3, 2])

# Note that if we were to use the weight matrices for model training,
# we would set requires_grad=True to update these matrices during training

# Compute the query, key and value vectors:
query_2 = x_2 @ W_query # 1x3 @ 3x2
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2.shape) # torch.Size([2])
print(query_2)
# tensor([0.4306, 1.4551])

# To obtain all keys and values:
keys = inputs @ W_key
values = inputs @ W_value
print('keys.shape:', keys.shape)
print('values.shape:', values.shape)

'''
keys.shape: torch.Size([6, 2])
values.shape: torch.Size([6, 2])
'''

# Compute the attention score ω22:
key_2 = keys[1]
attention_score_22 = query_2.dot(key_2)
print(attention_score_22) # tensor(1.8524)

# Computing all attention score given query 2
attention_score_2 = query_2 @ keys.T
print(attention_score_2)
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

# Computing attention weight
d_k = keys.shape[-1] # dimension of key
attention_weight_2 = torch.softmax(attention_score_2 / d_k**0.5, dim=-1)
print(attention_weight_2)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# Compute contect vector using matrix multiplication
#                     1x6               6x2
context_vector_2 = attention_weight_2 @ values
print(context_vector_2) # tensor([0.3061, 0.8210])
