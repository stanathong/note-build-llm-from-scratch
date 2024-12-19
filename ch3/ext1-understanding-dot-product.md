# Understanding dot product
> * The dot product of two vectors yields a scalar value.
> * The dot product is a measure of similarity because it quantifies how closely two vectors are aligned.
> * A higher dot product indicates a greater degree of alignment or similarity between the vectors.
> * In the content of **self attention**, the dot product determines **the extent to which each element in a sequence focuses on or attends to any other element**.
> * The higher the dot product the igher the similarity and attention score beween two elements.

**Hands-on**
* Code: [code/understanding-dot-product.py](code/understanding-dot-product.py)

```
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

# dot product beween of inputs[0] and query (inputs[1])
query = inputs[1] # journey
res = 0
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print('res:', res) # tensor(0.9544)
print(torch.dot(inputs[0], query)) # tensor(0.9544)
```
