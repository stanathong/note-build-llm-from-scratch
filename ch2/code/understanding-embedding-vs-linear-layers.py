# Understanding the difference between embedding layers and linear layers
'''
* Embedding layers in PyTorch accomplish the same as linear layers that perform matrix multitplication
* The reason we use embedding layers is computational efficiency.
'''
import torch

print('Pytorch version:', torch.__version__) # Pytorch version: 2.5.1

# Using nn.Embedding
# ------------------

# Suppose we have the following 3 training examples representing token IDs in the LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix is max(tokenID) + 1
# If the largest token ID is 3, we want 3+1 rows i.e.
# token IDs: 0, 1, 2, 3
num_idx = max(idx)+1

# The desired embedding dimension is hyperparameter
out_dim = 5

# Use a constant number as random seed for reproducibility
torch.manual_seed(123)

# weights in the embedding layer are initialised with small random values
embedding = torch.nn.Embedding(num_idx, out_dim)

print('Embedding weight:\n', embedding.weight)

'''
tensor([[ 0.3374, -0.1778, -0.3035, -0.5880,  1.5810],
        [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015],
        [ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953]], requires_grad=True)
'''

# Use the embedding layers to obtain the vector representation of Token 1
print('Token ID = 1:', embedding(torch.tensor([1])))

'''
Token ID = 1:  tensor([[ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
       grad_fn=<EmbeddingBackward0>)
'''

# embedding(torch.tensor([1]) <-- This is simply a lookup for index 1

# Similary, we can use embedding layers to obtain the vector representation of Token Id 2
print('Token ID = 2:', embedding(torch.tensor([2])))

'''
Token ID = 2: tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315]],
       grad_fn=<EmbeddingBackward0>)
'''

# To convert all token Ids into the vector reprsentation
idx = torch.tensor([2,3,1])
print(idx)
print(embedding(idx))
'''
tensor([2, 3, 1])
tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
        [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
       grad_fn=<EmbeddingBackward0>)
'''

# Using nn.Linear

# The embedding layer accomplishes exactly the same as nn.Linear layer on a one-hot encoded representation.

# 1. Converting the token IDs into a one-hot representation

idx = torch.tensor([2,3,1])
onehot = torch.nn.functional.one_hot(idx)
print(idx) # tensor([2, 3, 1])
print(onehot)
'''
tensor([2, 3, 1])

token    0  1  2  3
tensor([[0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]])
'''

# 2. Initialise a Linear layer which carries out a matrix multiplication XW^T
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print('Linear weight:\n', linear.weight)

'''
tensor([[-0.2039,  0.0166, -0.2483,  0.1886],
        [-0.4260,  0.3665, -0.3634, -0.3975],
        [-0.3159,  0.2264, -0.1847,  0.1871],
        [-0.4244, -0.3034, -0.1836, -0.0983],
        [-0.3814,  0.3274, -0.1179,  0.1605]], requires_grad=True)
'''

'''
Note that: 
The linear layer in PyTorch is also initialized with small random weights; to 
directly compare it to the `Embedding` layer above, we have to use the same 
small random weights, which is why we reassign them here:
'''

# Assigns a transposed version of the embedding.weight to the weights of a linear layer 
# (linear.weight). 
linear.weight = torch.nn.Parameter(embedding.weight.T)
print('linear.weight after reassignment:\n', linear.weight)

'''
tensor([[ 0.3374,  1.3010,  0.6957, -2.8400],
        [-0.1778,  1.2753, -1.8061, -0.7849],
        [-0.3035, -0.2010, -1.1589, -1.4096],
        [-0.5880, -0.1606,  0.3255, -0.4076],
        [ 1.5810, -0.4015, -0.6315,  0.7953]], requires_grad=True)
'''

# Now we can use the linear layer on the one-hot encoded representation

# Comparison between using linear layer and one hot with embedding
print(linear(onehot.float()))
print(embedding(idx))

'''
tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
        [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]], grad_fn=<MmBackward0>)

tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
        [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
'''




