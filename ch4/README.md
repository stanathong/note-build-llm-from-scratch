# Chapter 4: Implementing a GPT model from scratch to generate text

* In this chapter we will focus on
    * Coding a GPT-like LLM
    * Implementing transformer blocks 

<img width="771" alt="image" src="https://github.com/user-attachments/assets/e76b8eb8-9faa-488b-ac92-d4fe946e5225" />

## 4.1 Coding an LLM architecture

* LLMs such as GPT (which stands for generative pretrained transformer) are large deep neural network architectures designed to generate new text one word (or token) at a time.

<img width="637" alt="image" src="https://github.com/user-attachments/assets/fc2dad7f-9d16-493a-98fc-0c4ab627897e" />

* The picture above provides a top-down view of a GPT-like LLM.
* What we have done in the previous chapters:
    * Input tokenization
    * Input embedding
    * Masked multi-head attention module 
* In this step, we will implement the core structure of the GPT model including its transformer blocks, which will be trained to generate human-like text.
* In the concept of deep learning and LLMs like GPT, the term "parameters" refer to the trainable weights of the model.
* These weights are the internal variables of the model that are adjusted and optimised during the training process to minimise a specific loss function.
* This optimisation allows the model to learn from the training data.
* We specify the configuration of the small GPT-2 model via the Python dictionary below:

```
GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}
```

* The variables defined in the GPT_CONFIG_124M are
* vocab_size: a vocabulary of 50,257 words as used by the BPE tokenizer
* context_length: the max. number of input tokens the model can handle via the positional embedding
* emb_dim: the embedding size which transforms each token into a 768-dimensional vector
* n_heads: the number of attention heads in the multi-head attention
* n_layers: the number of transformer blocks in the model.
* drop_rate: the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting
* qkv_bias: whether to include a bias vector in the Linear layers of the multi-head attention for query, key and value, computations. 


* We wil use this configuration to implement a GPT placehoder architecture, called DummyGPTModel

<img width="607" alt="image" src="https://github.com/user-attachments/assets/95cf6f00-f83c-402a-9518-a382af2fd86c" />

* Code: [code/section-4.1-dummy-gpt-model.py](code/section-4.1-dummy-gpt-model.py)

* Dummy GPT Model
```
import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Recall that nn.Embedding layer is just a look up table for discrete indes
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        
        # one position per one token in the context
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        # Uses a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx):
        # batch size and context length
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # get token from the look up table
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# A placeholder that will be replaced by a real Transformer block
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    # A dummy forward function
    def forward(self, x):
        return x

# A placeholder that will be replaced by a real LayerNorm
class DummyLayerNorm(nn.Module):
    # normalized_shape: embedding_dim
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
```

* Caller implementation

```
tokenizer = tiktoken.get_encoding('gpt2')
batch = []

txt1 = 'Every effort moves you'
txt2 = 'Every day holds a'

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

print(batch.shape)
print(batch)

'''
torch.Size([2, 4])
tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])
'''

# Initialise a new 124M-param DummyGPTModel instance and feed the tokenised batch
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print(logits.shape)
print(logits)
```

* The output:

```
torch.Size([2, 4, 50257])
tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
         [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
         [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
         [ 0.0139,  1.6754, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],

        [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
         [-0.7860,  0.5581, -0.0610,  ...,  0.4835, -0.0077,  1.6621],
         [ 0.3567,  1.2698, -0.6398,  ..., -0.0162, -0.1296,  0.3717],
         [-0.2407, -0.7349, -0.5102,  ...,  2.0057, -0.3694,  0.1814]]],
       grad_fn=<UnsafeViewBackward0>)
```
* `torch.Size([2, 4, 50257])`: the output tensor has 2 rows corresponding to 2 text sample, each sample consits of 4 tokens (context length). Each token has 50257-dim vector, which corresponding to the size of vocabulary.
* The post processing task will convert these 50257-dim vector into token IDs, which can be decoded into words.


<img width="632" alt="image" src="https://github.com/user-attachments/assets/81bbdd5a-6b56-44c2-8bd9-4bca1ffde0f7" />

## 4.2 Normalizing activations with layer normalization

* *later normalization* is used to imporove the stability and efficiency of nn training.
* **The key idea:** to adjust the activations (outputs) of an nn layer to have a mean of 0 and variance of 1 (known as unit variance).
* This adjustment speeds up the convergence to effective weights and ensure consistent, reliable training.
* In GPT2 and modern transformer architectures, layer normalization is typically applied before and after the multi-head attention module.

<img width="619" alt="image" src="https://github.com/user-attachments/assets/d37c57a0-6e6d-4292-a62f-c4a0cfe934c6" />

* Below is to recreate the example above, in which we have a NN layer with 5 inputs and 6 outputs.

```
import torch
import torch.nn as nn
import tiktoken

torch.manual_seed(123)

# Create 2 training samples with 5 dimensions/features each
batch_example = torch.randn(2, 5)
print(f'batch_example:\n{batch_example}')

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(f'output:\n{out}')
```

```
batch_example:
tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
        [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]])
output:
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
```

* Compute the mean and variance of the output tensor

# Compute its means and variance

```
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')
```

* The output's mean and variance:
```
mean: tensor([[0.1324],
        [0.2170]], grad_fn=<MeanBackward1>)
var: tensor([[0.0231],
        [0.0398]], grad_fn=<VarBackward0>)
```

* Apply layer norm to the outputs: (output - mean)/sqrt(var).

```
out_norm = (out - mean) / torch.sqrt(var)
print(f'out_norm:\n{out_norm}')
```

* The output is:

```
out_norm:
tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
       grad_fn=<DivBackward0>)
```

* Recompute mean and variance again

```
# Recompute mean and variance again
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')
```

* Output which we expect it to have mean = 0, and variance = 1 (which is very close)

```
mean: tensor([[-5.9605e-08],
        [ 1.9868e-08]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
```

* We can turn off the scientific notation to improve readability.

```
torch.set_printoptions(sci_mode=False)
print(f'mean: {mean}')
print(f'var: {var}')

'''
mean: tensor([[    -0.0000],
        [     0.0000]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
'''
```

* Implement a layer normalization class

* Code: [code/section-4.2-layer-normalization-class.py](code/section-4.2-layer-normalization-class.py)
```
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # In the variance calculation, we devide by n instead of n-1.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```
* The layer normalization class operates on the last dim of the input tensor x, which represents the embedding dimension.
* The scale and shift are trainable parameters. It allows the model to learn appropriate scaling and shifting that best suite the data it is processing.

* Test the LayerNorm class with the example.

```
torch.manual_seed(123)

# Create 2 training samples with 5 dimensions/features each
batch_example = torch.randn(2, 5)
print(f'batch_example:\n{batch_example}')

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
# Compute mean and variance
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print(f'mean: {mean}')
print(f'var: {var}')
```

* Output:

```
mean: tensor([[1.0000],
        [1.0000]], grad_fn=<MeanBackward1>)
var: tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
```

**Layer normalization vs batch normalization**

> Batch normalization normalizes across the batch dimension, while layer normalization normalizes across the feature dimension.\
> Batch size depends on computational resources.\
> Layer normalization normalizes each input independently of the batch size, it offers more flexibility and stability in these scenarios.

## 4.3 Implementing a feed forward network with GELU activations

* ReLU has been commonly used in DL due to its simplicity and effectiveness.
* In LLMs, several activation functions are employed. The two notable examples are
    * GELU (Gaussian error linear unit)
    * SwiGLU (Swish-gated linear unit)
* They improve performance of DL models.

### GELU

* GELU is defined as

```
GELU(x) = x⋅Φ(x)

where Φ(x) is the cumulative distribution func- tion of the standard Gaussian distribution.
```

* In practice, it's common to implement a computationally cheaper approximation:

<img width="512" alt="image" src="https://github.com/user-attachments/assets/6850a46c-2198-4272-9158-50bc0f6938f6" />

* Code: [code/section-4.3-GELU-activation-function.py](code/section-4.3-GELU-activation-function.py)

```
import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))) 
        )

gelu, relu = GELU(), nn.ReLU()
```

* Plot the results comparing GELU and ReLU

```
import matplotlib.pyplot as plt

# Create 100 sample data points in the range -3 to 3
x = torch.linspace(-3,3,100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'RELU']), 1):
    plt.subplot(1,2,i)
    plt.plot(x, y)
    plt.title(f'{label} activation function')
    plt.xlabel('x')
    plt.ylabel(f'{label}(x)')
    plt.grid(True)
plt.tight_layout()
plt.show()
```

<img width="788" alt="image" src="https://github.com/user-attachments/assets/57e1ce11-1f6e-443f-abc7-26274868eec2" />

* GELU is a smooth, non-linear function that approximates ReLU but with a non zero gradeint for almost all negative values (except at approximately x = -0.75)
* ReLU is a piecewise linear function.
* The smoothness of GELU allows for more nuanced adjustment to the model's parameters.
* Unlike ReLU which output zero for any negative input, GELU allows for a small, non-zero output for negative values.
* This characteristic allows neurons that receive negative input to continue contributing to the learning process, although to a less extent than positive inputs.

### Feed Forward Neural Network Module

* Code: [code/sectino-4.3-FeedForward.py](code/sectino-4.3-FeedForward.py)
```
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), # 768 x 3072
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']), # 3072 x 768
        )
    
    def forward(self, x):
        return self.layer(x)
```

```
ffn = FeedForward(GPT_CONFIG_124M)
# Create a sample input of batch size = 2, 3 tokens, token embedding size = 768
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape) # torch.Size([2, 3, 768])
```

<img width="487" alt="image" src="https://github.com/user-attachments/assets/f48a4c08-dcee-48d7-b311-7e9f3b897df1" />

> Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space through the first linear layer. This expansion is followed by a nonlinear GELU activation and then a contraction back to the original dimension with the second linear transformation.\
> **Such a design allows for the exploration of a richer representation space.**

<img width="621" alt="image" src="https://github.com/user-attachments/assets/0b1987fd-f1e5-4fb5-8f50-ba7077138782" />

## 4.4 Adding shortcut connections

### Concepts behind shortcut connections, also known as skip or residual connections

* Shortcut connections are used in deep networks to mitigate the challenge of vanishing gradients.
* This is the problem where gradients (guide weight updates during training) become progressively smaller as they propate backward through the layers - making it difficult to train the earlier layers.

<img width="748" alt="image" src="https://github.com/user-attachments/assets/17970d8c-6949-421f-a323-e95cc7b4ec88" />

* The shortcut connections skip one or more layers, and add the output of one layer to the output of a later layer.
* This is crucial in preserving the flow of gradients during the backward pass in the training.

**Hands-on**
* Code: [code/section-4.4-ExampleDeepNeuralNetwork.py](code/section-4.4-ExampleDeepNeuralNetwork.py)

```
import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # Implement 5 layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
                          GELU()),                                                    
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Compute output of the current layer
            layer_output = layer(x)
            print(f'layer#{i} : x = {x.shape}, output = {layer_output.shape}')
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    '''
    layer#0 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#1 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#2 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#3 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#4 : x = torch.Size([1, 3]), output = torch.Size([1, 1])
    '''
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))) 
        )

# Utility function to compute gradients in the model's backward pass
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    # calculate loss based on how close the target and the output
    loss = nn.MSELoss()
    loss = loss(output, target)
    # calculate gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')
```

* **Without shortcut connections**
```
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0, -1.]])

# Without shortcut connections
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)
```
* **Output when not using shortcust connections**
    * Gradient becomes smaller as we progress from the last layer (4) to the first layer (0)

```
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.00012011159560643137
layers.2.0.weight has gradient mean of 0.0007152039906941354
layers.3.0.weight has gradient mean of 0.0013988736318424344
layers.4.0.weight has gradient mean of 0.005049645435065031
```

* **With shortcut connections**
```
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0, -1.]])

# With shortcut connections
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_without_shortcut, sample_input)
```
* **Output with shortcut connections**
    * Gradient value stabilizes as we progress toward the first layer (layers.0) and doesn't shrink to a vanishingly small value.

```
layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694106817245483
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732204914093
layers.4.0.weight has gradient mean of 1.3258540630340576
```

> **Note:**
> nn.ModuleList does not have a forward method, but nn.Sequential does.\
> We can wrap several modules in nn.Sequential and run it on the input.\
> nn.ModuleList is just a Python list that stores a list nn.Modules and it does not have a forward() method. It can not be called like a normal module.

## 4.5 Connecting attention and linear layers in a transformer block

* In this section, we're implementing the transformer block.
* Note that this block is repeated 12 times in the GPT2-architecture.

<img width="712" alt="image" src="https://github.com/user-attachments/assets/d153828e-efef-4584-aa57-5c086110610a" />

* Above is the transformer block.
* It takes input tokens which have been embedded into 768 dimensional vector.
* Each row reprsents one token's vector representation.
* The output of transformer block are vectors of the same dimension as the input.

* **Hands-On**

* Code: [code/transformer_block.py](code/transformer_block.py)

```
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from components_block import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

* Code: [code/section-4.5-transformer-block.py](code/section-4.5-transformer-block.py)

```
import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from config import GPT_CONFIG_124M

torch.manual_seed(123)

# Create sample input of shape: [batch_size, num_tokens, emb_dim]
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape) # torch.Size([2, 4, 768])
print("Output shape:", output.shape) # torch.Size([2, 4, 768])
```

* The transformer block maintains the input dimensions in its output, indicating that the transformer architecture processes sequences of data without altering their shape throughout the network.
* Based on this design, each output vector directly corresponds to an input vector, maintaining a one-to-one relationship.
* The output (context vector) encapsulates information from the entire input sequence.
* This means that while the physical dimensions of the sequence (length and feature size) remain unchanged as it passes through the transformer block, **the content of each output vector is re-encoded to integrate contextual information from across the entire input sequence**.

## 4.6 Coding the GPT model

**An overview of the GPT model architecture showing the flow of data through the GPT model**
* Startint from the bottom to the top:
    * Input text -> tokenized text
    * tokenized text -> token embedding
    * token embedding -> augmented with positional embeddings
    * the tensor -> transformer blocks 
<img width="735" alt="image" src="https://github.com/user-attachments/assets/ab0646e3-c31b-4a9b-9fc1-d69e5c2dffb3" />

* In the case of the 124-million-parameter GPT-2 models, the transformer block is repeated 12 times, as specified in the `n_layers` entry in the `GPT_CONFIG_124M` dictionary.
* In the 1,542-million-parameter GPT-2 model, the transformer block is repeated 48 times.
* The output from the final transformer block then goes through `Final LayerNorm` and later the `Linear output layer`.
* The last linear output layer maps each token vector into a 50257 dimentional embedding, which is equal to the model's vocabulary size, to predict the next token in the sequence.

* **Hands-on**

* Code: [code/gpt_model.py](code/gpt_model.py)
```
import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from components_block import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x) # same shape as input
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

* The `__init__` constructor initialises the token and positional embedding layers using the configuration named cfg below:

```
GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}
```

* These embedding layers converts input token indices into dense vectors and adding positional information.
`x = tok_embeds + pos_embeds`.
* The `__init__` method creates a sequential stack of `cfg["n_layers"]` TransformerBlock models.
* Following the transformer blocks, a LayerNorm layer is applied.
* This LayerNorm is used to standardizing the outputs from the transformer blocks to stabilize the learning process.
* Finally a linear output head is used to proejcts the transformer's output into the vocabulary space of the tokenizer to generate logits for each token in the vocabulary.

* **Testing the 124-million-parameter GPT model**
* We start by initialise the 124-million-parameter GPT model using the GPT_CONFIG_124M dictionary.
* We then feed to the model the batch text input.

* Code: [code/section-4.6-gpt-model.py](code/section-4.6-gpt-model.py)
```
import torch
import torch.nn as nn
from gpt_model import GPTModel 

torch.manual_seed(123)

# Input is tokenised text
batch = torch.tensor([
        [6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print('Input batch:', batch.shape, '\n')
print('Input batch:\n', batch)

print('\nOutput shape:', out.shape)
print(out)
```

* The tokenized input is

```
Input batch: torch.Size([2, 4]) 

Input batch:
 tensor([[6109, 3626, 6100,  345], # Token IDs of text 1
        [6109, 1110, 6622,  257]]) # Token IDs of text 2
```

* The output obtained from the GPTModel is

```
Output shape: torch.Size([2, 4, 50257])
tensor([[[ 0.4398, -1.1968, -0.3533,  ..., -0.1638, -1.2250,  0.0803],
         [ 0.1247, -2.2218, -0.6962,  ..., -0.5499, -1.4728,  0.0665],
         [ 0.5515, -1.5762, -0.3643,  ...,  0.0276, -1.7843, -0.2937],
         [-0.8036, -1.6966, -0.2890,  ...,  0.3314, -1.2682,  0.1784]],

        [[-0.3290, -1.8522, -0.1652,  ..., -0.1751, -1.0380, -0.2999],
         [-0.0083, -1.2779, -0.1241,  ...,  0.3117, -1.4347,  0.2552],
         [ 0.5651, -1.1005, -0.1858,  ...,  0.1592, -1.2875,  0.2329],
         [-0.5593, -1.3399,  0.3970,  ...,  0.8095, -1.6276,  0.3201]]],
       grad_fn=<UnsafeViewBackward0>)
```

* The output tensor has the shape `[2, 4, 50257]`, since we passed in two input texts (batch = 2) with four tokens each.
* The last dimension (50,257) corresponds to the vocabulary size of the tokenizer.

* **Computing the number of parameters**
```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Total number of parameters: 163,009,536
```

* **Explanation why the actual number of parameters is 163M instead of 124M**

> This involes a concept called **weight tying**, which was used in the original GPT-2 architecture.
> The original GPT-2 architecture **reuses the weights from the token embedding layer in its output layer**.
> To understand better, have a look at the shapes of the **token embedding layer and linear output layer**:

```
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
```
* The weight tensors for both of these layers have the same shape which is `[50257, 768]`.

```
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])
```
* The token embedding and output layers are very large due to the number of rows for the 50,257 in the tokenizer’s vocabulary.
* If we remove the output layer parameter count from the total GPT-2 model count according to the weight tying:

```
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}")
```

```
Number of trainable parameters considering weight tying: 124,412,160
```

* The memory required for the 163-million parameters can be computed as:
    * Calculate the total size in bytes, assuming float32 (4 bytes per parameters)
    * Convert to megabytes by dividing with 10^6
```
total_size_bytes = total_params * 4 # assuming float32 i.e. 4 bytes
total_size_mb = total_size_bytes / (1024*1024) # convert to MB

print(f"Total size of the model: {total_size_bytes:,} B or {total_size_mb:.2f} MB")
# Total size of the model: 652,038,144 B or 621.83 MB
```

* **Note:**
    * Weight tying reduces the memory footprint and computation of the model.
    * The author, however, suggest using separate token embedding and output layers as it results in better training and model performance.

## Exercise 4.1 Number of parameters in feed forward and attention modules
> Calculate and compare the number of parameters that are contained in the feed forward module
> and those that are contained in the multi-head attention module.

* Code: [code/exercise-4.1.py](code/exercise-4.1.py)
```
from transformer_block import TransformerBlock

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

block = TransformerBlock(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in block.ff.parameters())
print(f"The total number of parameters in feed forward module is {total_params:,}")
# The total number of parameters in feed forward module is 4,722,432

total_params = sum(p.numel() for p in block.att.parameters())
print(f"The total number of parameters in attention module is {total_params:,}")
# The total number of parameters in attention module is 2,360,064
```

* The outputs are:
```
The total number of parameters in feed forward module is 4,722,432
The total number of parameters in attention module is 2,360,064
```
* Note that this is the result per a single transformer block.
* Our GPT model consists of 12 transformer blocks.

## Excercise 4.2 Initializing larger GPT models
> We initialized a 124-million-parameter GPT model (GPT-2 small).\
> Without making any code modifications besides updating the configuration file,
> use the GPTModel class to implement\
> GPT-2 medium (using 1,024-dimensional embeddings, 24 transformer blocks, 16 multi-head attention heads),\
> GPT-2 large (1,280-dimensional embeddings, 36 transformer blocks, 20 multi-head attention heads),\
> and GPT-2 XL (1,600-dimensional embeddings, 48 transformer blocks, 25 multi-head attention heads).\
> As a bonus, calculate the total number of parameters in each GPT model.

- **GPT2-small** (the 124M configuration we already implemented):
    - "emb_dim" = 768
    - "n_layers" = 12
    - "n_heads" = 12

- **GPT2-medium:**
    - "emb_dim" = 1024
    - "n_layers" = 24
    - "n_heads" = 16

- **GPT2-large:**
    - "emb_dim" = 1280
    - "n_layers" = 36
    - "n_heads" = 20

- **GPT2-XL:**
    - "emb_dim" = 1600
    - "n_layers" = 48
    - "n_heads" = 25

* Code: [code/exercise-4.2.py](code/exercise-4.2.py)
```
from gpt_model import GPTModel

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

def get_config(base_config, model_name='gpt2-small'):
    GPT_CONFIG = base_config.copy()

    if model_name == 'gpt2-small':
        GPT_CONFIG['emb_dim'] = 768
        GPT_CONFIG['n_layers'] = 12
        GPT_CONFIG['n_heads'] = 12

    elif model_name == 'gpt2-medium':
        GPT_CONFIG['emb_dim'] = 1024
        GPT_CONFIG['n_layers'] = 24
        GPT_CONFIG['n_heads'] = 16

    elif model_name == 'gpt2-large':
        GPT_CONFIG['emb_dim'] = 1280
        GPT_CONFIG['n_layers'] = 36
        GPT_CONFIG['n_heads'] = 20

    elif model_name == 'gpt2-xl':
        GPT_CONFIG['emb_dim'] = 1600
        GPT_CONFIG['n_layers'] = 48
        GPT_CONFIG['n_heads'] = 25

    else:
        raise ValueError(f'Incorrect model name {model_name}')
    
    return GPT_CONFIG

def calculate_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters: {total_params:,}")

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    
    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"Total size of the model: {total_size_mb:.2f} MB")

if __name__ == '__main__':
    for model_abbrev in ("small", "medium", "large", "xl"):
        model_name = f"gpt2-{model_abbrev}"
        CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name)
        model = GPTModel(CONFIG)
        print(f"\n\n{model_name}:")
        calculate_size(model)
```

* Output:

```
gpt2-small:
The total number of parameters: 163,009,536
Number of trainable parameters considering weight tying: 124,412,160
Total size of the model: 621.83 MB


gpt2-medium:
The total number of parameters: 406,212,608
Number of trainable parameters considering weight tying: 354,749,440
Total size of the model: 1549.58 MB


gpt2-large:
The total number of parameters: 838,220,800
Number of trainable parameters considering weight tying: 773,891,840
Total size of the model: 3197.56 MB


gpt2-xl:
The total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

## 4.7 Generating text

* To recall, a GPT model generates generates text, one token at a time.
* Starting with an initial context, the model predicts a subsequent token during each iteration, and appends it to the input for the next round of predictions.

```
               Input     [Output]
Iteration 1:   Hello, I am [a] <-- the output token will append to the input text and becomes the input for the next iteration.
Iteration 2:   Hello, I am a [model]
Iteration 3:   Hello, I am a model [ready]
```

* Our current GPT model implementation outputs tensors with shape `[batch_size, num_token, vocab_size]`.
* The next step is to generate text from these output tensors.

<img width="688" alt="image" src="https://github.com/user-attachments/assets/5058fffe-26dd-4b6f-b992-e7fd80cbb00d" />

* This process of generating the next token is repeated over many iterations, until we reach a user-specified number of generated tokens.
* The following `generate_text_simple` function implements **greedy decoding**, which is a simple and fast method to generate text
* In greedy decoding, at each step, **the model chooses the word (or token) with the highest probability as its next output** (the highest logit corresponds to the highest probability, so we technically wouldn't even have to compute the softmax function explicitly)

* **Hands-on**

* Code: [code/section-4.7-generate-text-sample.py](code/section-4.7-generate-text-sample.py)
```
import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    '''
    idx [batch, n_tokens]: array of indices in the current context
    '''
    # Loop for # max_new_tokens required to generate
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # keep the last tokens as context
        idx_cond = idx[:, -context_size:] # [batch, context_size]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # -1 here indicates the last token 
        logits = logits[:, -1, :] # [batch, vocab_size]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1) # [batch, vocab_size]

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=-1) # (batch, n_tokens + 1)

    return idx
```
* We obtain logits as predictions from the model.
* We then use a softmax function to convert the logits into a probablity distribution, from which we identify the position with the highest value via torch.argmax.
* The model generates the most likely next token, which is known as **greedy decoding**.

* We first start by encode the input text into token IDs:

```
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
print("encoded_tensor.shape:", encoded_tensor.shape)
```

* The encoded IDs are:
```
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])
```

* Next, we initialise the model and put it into the eval() mode to disable random components like dropout() which are only used during training.

```
from gpt_model import GPTModel 
from config import GPT_CONFIG_124M

# Put the model into .eval() mode
# This disable random components like dropout, which are only used during training.
model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# Convert the IDs back into text using the tokenizer
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
```

* The output we obtained are:

```
Output: tensor([[15496,    11,   314,   716,  1755, 31918,  8247,  1755, 37217,  4085]])
Output length: 10
Hello, I am night Rafaelzens night travellers emot
```
* The model generates gibberish because we haven't trained the model yet.
* We just initialise the model with initial random weights.

## Excercise 4.3 Using separate dropout parameters

* **Code:** [code/excercise-4.3.py](ch4/code/excercise-4.3.py)

### New config

* Instead of using a single `dropout=cfg["drop_rate"]` setting in the GPT_CONFIG_124M dictionary, we change the code to specify separate dropout for different modules.

* Originally, it is defined as:

```
GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}
```

* It is now currenly defined as:

```
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,        # NEW: dropout for embedding layers
    "drop_rate_attn": 0.1,       # NEW: dropout for multi-head attention  
    "drop_rate_shortcut": 0.1,   # NEW: dropout for shortcut connections  
    "qkv_bias": False
}
```

* Use the drop-rate settings for multi-head attention for TransformerBlock:
* There are used in two places.

```
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate_attn"], # NEW: dropout for multi-head attention
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"]) # NEW: dropout for shortcut connections

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```

* Use the drop_rate setting in GPTModel:

```
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"]) # NEW: dropout for embedding layers

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

## Summary

* **Layer normalization stabilizes training** by ensuring that each layer’s outputs have a consistent mean and variance.
* Shortcut connections skip one or more layers by feeding the output of one layer directly to a deeper layer, which helps mitigate the vanishing gradient problem when training deep neural networks, such as LLMs.
* Transformer blocks are a core structural component of GPT models, combining masked multi-head attention modules with fully connected feed forward networks that use the GELU activation function.
* GPT models are LLMs with many repeated transformer blocks that have millions to billions of parameters.
* GPT models come in various sizes, for example, 124, 345, 762, and 1,542 million parameters, which we can implement with the same GPTModel Python class.
* The text-generation capability of a GPT-like LLM involves decoding output tensors into human-readable text by sequentially predicting one token at a time
based on a given input context.
* Without training, a GPT model generates incoherent text, which underscores the importance of model training for coherent text generation.
