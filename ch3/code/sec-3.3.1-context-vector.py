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
print('inputs:\n', inputs)

# Compute attention scores between the query token x(2)
# and all other input token.
num_tokens = inputs.shape[0]
query = inputs[1] # journey

# compute attention score by dot product of each input with query
attention_score_idx = torch.empty(num_tokens)
for i, x_i in enumerate(inputs):
    attention_score_idx[i] = torch.dot(x_i, query)

print(attention_score_idx)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# Normalise the attention score
attention_score_sum = attention_score_idx.sum()
attention_weights = attention_score_idx / attention_score_sum
print('attention_score:', attention_score_idx)
print('attention_weights:', attention_weights)
print('sum of attention_score:', attention_score_sum)
print('sum of attention_weight:', attention_weights.sum())

'''
attention_score: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
attention_weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
sum of attention_score: tensor(6.5617)
sum of attention_weight: tensor(1.0000)
'''

def softmax_custom(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Use softmax for normalisation instead
attention_weights_custom = softmax_custom(attention_score_idx)
print('attention_weights_custom:', attention_weights_custom)
print('sum of attention_weights_custom:', attention_weights_custom.sum())

'''
attention_weights_custom: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
sum of attention_weights_custom: tensor(1.)
'''

# Use pytorch softmax
attention_weights_softmax = torch.softmax(attention_score_idx, dim=0)
print('attention_weights_softmax:', attention_weights_softmax)
print('sum of attention_weights_softmax:', attention_weights_softmax.sum())

'''
attention_weights_softmax: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
sum of attention_weights_softmax: tensor(1.)
'''

# Calculating the context vector 
query = inputs[1]
context_vector_idx1 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_idx1 += attention_weights_softmax[i] * x_i

print('context_vector_idx_1:', context_vector_idx1)
# context_vector_idx_1: tensor([0.4419, 0.6515, 0.5683])
