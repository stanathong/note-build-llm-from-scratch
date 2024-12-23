# Chapter 3: Coding attention mechanisms

## Introduction

* In this chapter, we will focus on attention mechanism, which is an important building block of GPT-like LLMs.

<img width="734" alt="image" src="https://github.com/user-attachments/assets/ab9d1b09-2fa8-4ed3-9fda-02b5f2c699b2" />

* We will implement 4 different variants of attention mechanism, each of which build on each other.
* The goal is to arrive at a compact and efficient implementation of multi-head attention that we can then plug into the LLM architecture.

<img width="666" alt="image" src="https://github.com/user-attachments/assets/d488fce9-95a4-40d2-8698-11588d5786cd" />

## 3.1 The problem with modeling long sequences

* We'll discuss the problem with pre-LLM architecture - that do not include attention mechanisms.
* Suppose we want to develop a language translation model that translates text from one language into another. We can't simply translate a text word by word due to the grammatical structures in the source and target language.

<img width="677" alt="image" src="https://github.com/user-attachments/assets/4c4f5aad-11df-49d7-b68e-b400d4a85341" />

* **To address this problem, it's common to use a deep neural network with two submodules: an encoder and a decoder. The encoder first read in and process the entire text, and the decoder then translate the text.**

* Before transformer Recurrent Neural Networks (RNNs) were the most popular encoder-decoder architecture from language translation.
* In RNN, output from previous steps are fed as inputs to the current step, making it well-suited for sequential data like text.
* In an encoder-decoder RNN, the input text is fed into the encoder, which processes it sequentially.
* The encoder updates its hidden state (the internal values at the hidden layers) at each step, trying to capture the entire meaning of the input sentence in the final hidden state.
* The decoder then takes this final hidden state to start generating the translated sentence, one word at a time. It also updates its hidden state at each step, which is supposed to carry the context neccessary for the next word prediction.

<img width="672" alt="image" src="https://github.com/user-attachments/assets/dfcfc86f-b42b-41c0-aa34-8e1cdf147a96" />

* The key-idea for the encoder-decoder RNNs is that
    * the encoder part processes the entire input text into a hidden state (memory cell).
    * the decoder takes in this hidden state to produce the output.  
* **The limitation of this architecture is that RNN can't directly access earlier hidden states from the encoder during the decoding phase. Consequently, it relies solely on the current hidden state. This can lead to a loss of context especially in complex sentences where dependencies might span long distances.**

## 3.2 Capturing data dependencies with attention mechanisms

* RNNs work fine for translating short sentences, they don't work well for longer texts as they don't have direct access to previous words in the input. (Look at the previous pic - we'll see that we can only access the latest node).
* The Bahdanau attention mechanism for RNNs modifies the encoder-decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step. 

<img width="676" alt="image" src="https://github.com/user-attachments/assets/abbfcbab-7484-4e46-be83-8de372afd0fc" />

* Then the transformer architecture was proposed later.
* **Self-attention allows each position in the input sequence to consider the relevancy of or attend to all other positions in the same sequence when computing the representation of a sequence.**
* It is a key component of contemporary LLMs based on the transformer architecture.

<img width="511" alt="image" src="https://github.com/user-attachments/assets/45e38183-f925-4905-9369-dd75092e9ce8" />

## 3.3 Attending to different parts of the input with self-attention

* Self-attention is the core of every LLM based on transformer architecture.

**The *self* in self-attention**
> In self-attention, the **self** refers to its ability to compute attention weights
> by **relating different positions within a single input sequence**.
> It assesses and learns the relationships and dependencies between various parts of
> the input intself, such as words in a sequence or pixels in an image.

### 3.3.1 A simple self-attention mechanism without trainable weights

* We'll begin by implementing **a simplified variant of self-attention**. This is free from any trainable weights.
* The goal is to illustrate a few key concepts in self-attention before adding trainable weights. 

> * We have an input sequence of token embedding x(1), x(2), ..., x(T).
> * We would like to compute a context vector for each input element that combines info from all other input.
> * Each input's contribution to the output context vector (for a specific input) is based on **attention weights**.

<img width="550" alt="image" src="https://github.com/user-attachments/assets/e2880693-35be-4bb5-b4a2-d221fe5acce2" />

**Figure 3.7**
> * We have an input sequence x consisting of T elements represented as x(1),..,x(T).
> * This sequence typically represents text such as **a sentence that has already been transformed into token embeddings.**
> * For example, an input text is "Your journey starts with one step."
> * In this case, each element of the sequence such as **x(1) corresponds to a d-dimensional embedding vector representing a specific token**, like "Your".
> * In this example, d = 3 - the input vectors are represented as three-dimensional embeddings.

* In **self-attention**, **the goal is to calculate context vector z(i) for each element x(i) in the input sequence.**
* A context vector can be interpreted as an enriched embedding vector.

**Figure 3.7**
> * We would like to compute the context vector z(2) for the second input element x(2), which correspond to the token "journey".
> * The enhanced context vector z(2) is an embedding that contains information about x(2) and all other input elements x(1) to x(T).

* **Context vectors** play a crucial role in self-attention.
* **Their purpose is to create enriched representations of each element in an input sequence** (like a sentence) by incorporating information from all other elements in the sequence.
* This is essential in **LLMs which need to understand the relationship and relevance of words in a sentence to each other**.
* Later, we will add trainable weights that help an LLM learn to construct these context vectors so theat they are relevant for the LLM to generate the next token.

* **First step:** we will implement a simplified self-attention mechanism to compute these weights and the resulting context vector one step at a time.

**Hands-on**
* Consider the following input sentence which has already been embedded into three-dimensional vectors. (This is quite a small embedding dimension)

* Code: [code/sec-3.3.1-context-vector.py](code/sec-3.3.1-context-vector.py)
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
```

*  The first step of implementing self-attention is to compute **the intermediate values ω, which is referred to as attention scores**.
*  We would like to compute the intermediate attention scores between the query token i.e. x(2) and each input token: x(1),x(2),..,x(T).
*  We determine these scores by computing the dot product of the query x(2) with every other input token.

<img width="633" alt="image" src="https://github.com/user-attachments/assets/b6ffa345-ae08-4f3b-94d6-4b28304036d0" />

**Hands-on**
* Code: [code/sec-3.3.1-context-vector.py](code/sec-3.3.1-context-vector.py)
```
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
```

**Understanding dot product**
* Reference: [ext1-understanding-dot-product.md](ext1-understanding-dot-product.md)
> * The dot product of two vectors yields a scalar value.
> * The dot product is a measure of similarity because it quantifies how closely two vectors are aligned.
> * A higher dot product indicates a greater degree of alignment or similarity between the vectors.
> * In the content of **self attention**, the dot product determines **the extent to which each element in a sequence focuses on or attends to any other element**.
> * The higher the dot product the igher the similarity and attention score beween two elements.

* After we obtain the attention scores, we normalise them by the sum of all attention scores.
* The main goal behind the normalisation is to obtain the attention weights that sum up to 1.
* The normalisation is a convention that is useful for interpretation and maintaining training stability in an LLM.

<img width="666" alt="image" src="https://github.com/user-attachments/assets/b82cff57-3974-44d5-920e-54e84f2f3362" />

**Hands-on**
* Code: [code/sec-3.3.1-context-vector.py](code/sec-3.3.1-context-vector.py)

```
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
num_tokens = inputs.shape[0]
query = inputs[1] # journey

attention_score_idx = torch.empty(num_tokens)
for i, x_i in enumerate(inputs):
    attention_score_idx[i] = torch.dot(x_i, query)

# Normalise the attention score
attention_score_sum = attention_score_idx.sum()
attention_weights = attention_score_idx / attention_score_sum

print('attention_score:', attention_score_idx)
print('attention_weights:', attention_weights)
print('sum of attention_score:', attention_score_sum)
print('sum of attention_weight:', attention_weights.sum())
```
**Output:**
```
attention_score: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
attention_weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
sum of attention_score: tensor(6.5617)
sum of attention_weight: tensor(1.0000)
```

* It's more common and advisable to use the `softmax` function for normalisation.
* This approach is better at managing extreme values and offers more favorable gradient properties during training.

**Hands-on**
```
def softmax_custom(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Use softmax for normalisation instead
attention_weights_custom = softmax_custom(attention_score_idx)
print('attention_weights_custom:', attention_weights_custom)
print('sum of attention_weights_custom:', attention_weights_custom.sum())
```

**Output:**
```
attention_weights_custom: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
sum of attention_weights_custom: tensor(1.)
```

* As the output shows, the softmax function also meets the objective and normalises the attention weights such that they sum to 1.
* In addition, **the `softmax` function ensures that the attention weights are always postive**.
* This makes the output interpretable as probabilities ore relative importance, where higher weights indicate greater importance.
\
* However, this custom softmax implementation `softmax_custom(x)` may encounter numerical instability problems, such as overflow and underflow, when dealing with large or small input values.
* In practice, it's better to use Pytorch's softmax.

```
# Use pytorch softmax
attention_weights_softmax = torch.softmax(attention_score_idx, dim=0)
print('attention_weights_softmax:', attention_weights_softmax)
print('sum of attention_weights_softmax:', attention_weights_softmax.sum())
```

**Output:**
```
attention_weights_softmax: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
sum of attention_weights_softmax: tensor(1.)
```

* Now that we have computed the normalised attention weights, we are ready for the final step i.e. calculating the context vector z(2) by multiplying the embedded input tokens x(i) with the corresponding attention weights then summing the resulting vector.

<img width="653" alt="image" src="https://github.com/user-attachments/assets/474ff958-0e14-48ee-9d61-b454449cebc0" />

```
# Calculating the context vector 
query = inputs[1]
context_vector_idx1 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_idx1 += attention_weights_softmax[i] * x_i

print('context_vector_idx_1:', context_vector_idx1)
# context_vector_idx_1: tensor([0.4419, 0.6515, 0.5683])
```
* Next, we will generalise this procedure for computing context vectors to calculate all context vectors simultaneously.

### 3.3.2 Computing attention weights for all input tokens

* In this section, we will extend the previous implementation to calculate attention weights and context vectors for all inputs.

<img width="498" alt="image" src="https://github.com/user-attachments/assets/ab06bef6-0315-4c85-8a95-87b5dce1434c" />

    * The tokens on each row is the query word.
    * The token on the top is the input word that the query word is obtained attention weight for.
```
Query: 1:journey
                                   0:Your 1:journey 2:starts 3:with  4.one    5.step
attention_weights_softmax: tensor([0.1385, 0.2379,   0.2333, 0.1240, 0.1082, 0.1581])
```

* Now we're going to compute all context vectors instead of only one at a time.

<img width="506" alt="image" src="https://github.com/user-attachments/assets/2fda5039-dea6-49b5-aad8-822dc768e0a7" />

**Hands-on**
* Code: [code/sec-3.3.2-context-vector.py](code/sec-3.3.2-context-vector.py)

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

print('inputs.shape:', inputs.shape) # # torch.Size([6, 3])

# Compute attention scores for all input
num_tokens = inputs.shape[0]
# score is arrange as 2D confusion matrix
attention_scores = torch.empty(num_tokens, num_tokens)
for i, query in enumerate(inputs):
    for j, token in enumerate(inputs):
        # i: row, j: col
        attention_scores[i, j] = torch.dot(query, token)

print(attention_scores)
```

* The resulting attention scores (unnormalised):
```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```
* The above implementation use 2 for-loops which is slow. We can achieve the same results using matrix multiplication.

```
# Instead of using 2 for-loops, we can use matrix multiplication
attention_scores = inputs @ inputs.T
print(attention_scores)
```

* The resulting attention scores (unnormalised) using matrix multiplication:
```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```

* Next step, we need to normalise each row so that **the values in each row sum to 1**.

```
# Normalise the attention scores using softmax
attention_weights = torch.softmax(attention_scores, dim=-1)
print(attention_weights)
```

* The computed attention weights are:
```
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])

# Compute sum for each row
print(torch.sum(attention_weights, dim=-1))
# tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

# sum of first row and first column
print(torch.sum(attention_weights[0], dim=-1)) # tensor(1.0000)
print(torch.sum(attention_weights[:,:1], dim=0)) # tensor([0.9220])
```

* In the final step, we use these attention weights to compute all context vectors via matrix multiplication.

```
# Step#3 - Compute context vector
#                         6x6              6x3
all_context_vectors = attention_weights @ inputs
print(all_context_vectors)
```

* The output of Step#3 is the three dimensional context vector for each word.
```
tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])
```

* Now we have finished implementing a simple self-attention mechanism.
* Next, we're adding trainable weights to enable LLM to learn from data and to improve its performance on specific tasks.

## 3.4 Implementing self-attention with trainable weights

* In this section, we will implement the self-attention mechanism used in the original transformer architecture, the GPT models, most other popular LLMs.
* This self-attention mechanism is called **scaled dot-product attention**.
* Figure 3.13 shows how this self-attention mechanism fits into the broader context of implementing an LLM.

<img width="729" alt="image" src="https://github.com/user-attachments/assets/615fbbf3-fe41-4ee0-ae1f-aef295102990" />

* In the previous section, we compute context vectors as weighted sum over the input vectors specific to a certain input element.
* In this section, we will introduce weight matrices that are updated during training.
* These trainable weights are the key that enables the attention module inside the model to produce *good* context vectors.

### 3.4.1 Computing the attention weights step by step

* In this section, we will implement the self-attention mechanism by introducing three 3 trainable weight matrices: Wq, Wk, and Wv.
* These 3 matrices are used to proejct the embedded input tokens x(i) into query, key and value vectors.

<img width="724" alt="image" src="https://github.com/user-attachments/assets/7a78f986-6565-4f00-8414-89b0806d09b5" />

* In the previous section, we maually compute attention score --> attention weights, then applying attention weights to the input tokens to obtain the context vector. In this section, the attention weights will be obtained by training.
* We'll start by computing only one context vector, z(2), which in fact z(1) or 'journey' when using base-0 index.
* We'll start by defining a few variables:

<img width="617" alt="image" src="https://github.com/user-attachments/assets/8b593064-26f9-4f6e-8be9-1586651c5604" />

**Hands-on**
* Code: [code/sec-3.4.1-attention-weights.py](code/sec-3.4.1-attention-weights.py)

```
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
```

* Initialise the three weight matrices: Wq, Wk, Wv.
* Note that if we were to use the weight matrices for model training, we would set `requires_grad=True` to update these matrices during training.

```
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print(W_query.shape) # torch.Size([3, 2])
```

* Compute the query, key and value vectors:

```
query_2 = x_2 @ W_query # 1x3 @ 3x2
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2.shape) # torch.Size([2])
print(query_2)
```
* The output for the query results in a 2-dimensional vector - this is because we set the output embedding size to 2.
```
tensor([0.4306, 1.4551])
```

> **Weight paramters vs attention weights**
> * In the weight matrices W, the term *weight* is short for **weight parameters**, the values of a neural network that are optimised during training.
> * These weights are not the same as attention weights.
> * Attention weights determine the extent to which a context vector depends on the different parts of the input (i.e. to what extent the network focuses on different parts of the input).
> * In summary, **weight parameters are the fundamental, learned cofficient that define the network's connections, while attention weights are dynamic, context-specific values**.

* Even though our temporary goal is only to compute the one context vector, z(2), **we still require the key and value vectors for all input elements as they are involved in computing attention weights** with respect to the query q(2), as shown in Figure 3.14.
* We can obtain all keys and values via matrix multiplication as shown below:

```
# To obtain all keys and values:
keys = inputs @ W_key
values = inputs @ W_value
print('keys.shape:', keys.shape)
print('values.shape:', values.shape)
```
* We can tell from the outputs below that, we successfully projected the six input tokens from a three-dimensional onto a two-dimensional embedding space.
> * keys.shape: torch.Size([6, 2])
> * values.shape: torch.Size([6, 2])

* **The next step is to compute the attention scores.**

<img width="731" alt="image" src="https://github.com/user-attachments/assets/35f3cc39-a3fa-4a59-9766-2419f9988046" />

* **First**, let's compute the attention score ω22:

```
key_2 = keys[1]
attention_score_22 = query_2.dot(key_2)
print(attention_score_22) # tensor(1.8524)
```
* The result is the unnormalised attetion score: tensor(1.8524).

* **Second**, we can generalise this computation to all attention scores via matrix multiplication.
```
attention_score_2 = query_2 @ keys.T
print(attention_score_2)
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
```
* The result at index 1 (1.8524) is the same as the result computed before.

* **Next**, we want to go from the attention scores to attention weights (= the normalised attention weights).
    * We compute the attention weights by (1) scaling the attention scores and (2) using the softmax function.
    * However, now we **scale the attention scores by dividing them by the square root of the embedding dimension of the keys** (taking the square root is mathematically the same as exponentiating by 0.5).

<img width="434" alt="image" src="https://github.com/user-attachments/assets/2d4be5cd-47c9-4a39-be18-0fd8510de767" />

```
# Computing attention weight
d_k = keys.shape[-1] # dimension of key
attention_weight_2 = torch.softmax(attention_score_2 / d_k**0.5, dim=-1)
print(attention_weight_2)
```

* Notice the outputs from the attetion score (top row) and the attention weight (bottom row).
> tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
>\
> tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

<img width="724" alt="image" src="https://github.com/user-attachments/assets/d8304b4d-b154-40c4-aef0-5f34f8d3ecd0" />

> **The rational behind saled-dot product attention**
> \
> <img width="434" alt="image" src="https://github.com/user-attachments/assets/e67bc313-5e93-4e5a-93b2-23fd08116c60" />
> \
> **The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding small gradients**.
> For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them.
> As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero.
> These small gradients can drastically slow down learning or cause training to stagnate.
> \
> The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

* The **final step** is to compute the context vectors, as illustrated in the picture below.

<img width="748" alt="image" src="https://github.com/user-attachments/assets/262e72ca-a878-4c7f-b02a-c48e0192cc29" />

* In the previous section, we compute the context vector by **multiplying attention weights to all the input embedding vector** and sum them to create the output context vector.
* In this section, we compute the context vector as a **weighted sum over the `value vectors`**.
* The attention weights now serve as a weighting factor that weight the respective important of each value vector.
* We can again use matrix multiplication to obtain the output in one step.

```
# Compute contect vector using matrix multiplication
#                     1x6               6x2
context_vector_2 = attention_weight_2 @ values
print(context_vector_2) # tensor([0.3061, 0.8210])
```

* So far, we've only computed a single context vector z(2). Next, we will generalise the code to compute all context vectors in the input sequence, z(1) to z(T).

> **Why query, key, and value?**
> * The terms “key,” “query,” and “value” in the context of attention mechanisms are borrowed from the domain of information retrieval and databases, where similar concepts are used to store, search, and retrieve information.
> * **A query is analogous to a search query in a database.** It represents *the current item (e.g. a word or token in a sentence) the model focuses on* or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention to pay to them.
> * **The key is like a database key used for indexing and searching.** In the attention mechanism, each item in the input sequence (e.g., each word in a sentence) has an associated key. These keys are used to match the query.
> * **The value in this context is similar to the value in a key-value pair in a database**. It represents the actual content or representation of the input items. *Once the model determines which keys (and thus which parts of the input) are most relevant to the query (the current focus item), it retrieves the corresponding values*.


### 3.4.2 Implementing a compact self-attention Python class

* **A compact self-attention Python class**
    * The __init__ method initialises the trainable weight matrices (W_query, W_key and W_value) for queries, keys, and values, each transforming the input dimension `d_in` to an output_dimension `d_out`.
    * During the forward pass, using the forward method,
    * 1) We compute the attention score (attention_scores) by multiplying queries and keys
    * 2) We then normalise the score using softmax and scale by the square root of its dimension.
    * 3) We create a context vector by weighting the values with these normalised attention scores.  

* Code: [code/sec-3.4.2-compact-self-attention-class.py](code/sec-3.4.2-compact-self-attention-class.py)
```
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
```
* We can use the class above the compute the context vector for our previous input example as

```
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
print('inputs.shape:', inputs.shape) # # torch.Size([6, 3])

d_in = inputs.shape[1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

# To compute the context vectors, we can do this by:
torch.manual_seed(123)

self_attention_v1 = SelfAttention_v1(d_in, d_out)
print(self_attention_v1(inputs))
```

* Since the inputs contains 6 embedding vectors, this results in an output of 6 context vectors:

```
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
```

* The picture below summarise the self-attention mechanism we have just implemented.

<img width="750" alt="image" src="https://github.com/user-attachments/assets/09ef07ad-0c87-4efe-86ab-00f39fa60c52" />

* Self-attention involves the trainable weight matrices: Wq, Wk, Wv.
* These 3 matrices: Wq, Wk, Wv transform input data into queries, keys and values.
* As the method is exposed to more data during training, it adjusts these trainable weights.
\
* We can improve the `SelfAttention_v1` class further by utilising Pytorch's nn.Linear layers which effectively perform matrix multiplication when the bias units are disabled.
* The advantance of using Pytorch's nn.Linear layers instead of nn.Parameter is hat nn.Linear has an optimised weight initialisation scheme, contributing to more stable and effective model training. 

* Code: [code/sec-3.4.2-compact-self-attention-class_v2.py](code/sec-3.4.2-compact-self-attention-class_v2.py)
```
import torch.nn as nn
import torch

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
```

* We can use the class to obtain the results as:

```
d_in = inputs.shape[1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

# To compute the context vectors, we can do this by:
torch.manual_seed(789) # *** different seed value

self_attention_v2 = SelfAttention_v2(d_in, d_out)
print(self_attention_v2(inputs))
```

* Note that the outputs we obtain below are different from the one we obtain from previous version, this is because we use different initial weights for the weight matrices since nn.Linear uses a more sophisticated wieght initialisation scheme.

```
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
```
* In the next step, we will make an enhancement to the self-attetion mechanism focusing specifically on incorporating **causal and multi-head elements**.
* **The causal aspect** involves modifying the attention mechanism to prevent the model from accessing future information in the sequence. This is essential for tasks like language modeling, where each word prediction should only depends on previous words.
* **The multi-head component** involves splitting the attention mechanism into multiple "heads". Each head learns different aspects of the data, allowing the model to simultaneously attend to information from different representation subspaces at different positions. This improves the model's performance in complex tasks.

### Excercise 3.1 Comparing SelfAttention_v1 and SelfAttention_v2

> `nn.Linear` in SelfAttention_v2 uses a different weight initialisation scheme than `nn.Parameter(torch.rand(d_in, d_out))` used in SelfAttention_v1, which causes both mechanisms to produce different results.
\
> To check that both implementations, SelfAttention_v1 and SelfAttention_v2, are similar, we can *transfer the weight matrices from a SelfAttention_v2 object to a Self- Attention_v1*, such that both objects produce the same results.
> **Task**: is to correctly assign the weights from an instance of SelfAttention_v2 to an instance of SelfAttention_v1.
> To do this, you need to understand the relationship between the weights in both versions. (Hint: nn.Linear stores the weight matrix in a transposed form.) After the assignment, you should observe that both instances produce the same outputs.

* Code: [code/excercise-3.1.py](code/excercise-3.1.py)

```
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
```

**The imporant change to make the same weights.**
```
# nn.Linear stores the weight matrix in a transposed form, hence transpose is required here.
self_attention_v1.W_query = torch.nn.Parameter(self_attention_v2.W_query.weight.T)
self_attention_v1.W_query = torch.nn.Parameter(self_attention_v2.W_query.weight.T)
self_attention_v1.W_value = torch.nn.Parameter(self_attention_v2.W_value.weight.T)

print('Output from v1:\n', self_attention_v1(inputs))
print('Output from v2:\n', self_attention_v2(inputs))
```

**Outputs:**
```
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
```

## 3.5 Hiding future words with causal attention

* For many LLM tasks, **we want the self-attion mechanism to consider only the tokens that appear prior to the current position** when predicting the next token in a sequence.
* Causal attention or Masked attention is a specialised form of self-attention.
* **Causal attention restrics a model to only consider previous and current inputs**.
* This is **in constrast** to the standard self-attention mechanism, which allows access to the entire input sequence at once!

> The key concept is to ensure that each token can **only attend to previous tokens and itself and NOT future tokens**.
> This constraint maintains the causality of the sequence, preventing the model from "cheating" by looking ahead at future data when making predictions.
> Causal attention typically involves **masking the attention scores** so that each element i in the sequence can only attend to elements 1, 2, .., i and not to any element j > i.

* Next, we're going to modify the standard self-attention mechanism to create a causal attention mechanism.
* To achieve this in GPT-like LLMs, for each token processed, we **mask out the future tokens**, which come after the current token in the input text.

> This can be implemented by modifying the attention mechanism with a triangular mask (upper triangular part is masked).
> This ensures that each token only attends to previous tokens and not the future ones.

<img width="675" alt="image" src="https://github.com/user-attachments/assets/a7c7a624-c791-4262-a1b6-e53468666a57" />

* We mask out the attention weights above the diagonal, and **normalise the non-masked attention weights such that the attention weights sum to 1 in each row**.

### 3.5.1 Apply a causal attention mask

* To implement the steps to apply a causal attention mask - in order to obtain the masked attention weights, see the fig. below.

<img width="678" alt="image" src="https://github.com/user-attachments/assets/af96897a-06d3-473b-81ee-deff240f99ac" />

* Overview steps:
    1) In: Attention scores (unnormalised) ----> Apply softmax ----> Out: Attention weights (normalised)
    2) In: Attention weights (normalised) ----> Mask with 0's above diagonal ----> Out: 0 for scores at upper-triangle (now become unnormalised)
    3) In: Masked attention scores (unnormalised) ----> Normalise row ----> Out: Masked attention weights (normalised)

* Code: [code/sec-3.5.1-causal-attention-mask.py](code/sec-3.5.1-causal-attention-mask.py)
```
import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
         pass
        
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
print('inputs.shape:', inputs.shape) # # torch.Size([6, 3])

d_in = inputs.shape[-1] # The input embedding size, d_in = 3
d_out = 2 # The output embedding size, d_out = 2

torch.manual_seed(789)

self_attention_v2 = SelfAttention_v2(d_in, d_out)
```
* **First**, compute the attention weights using the softmax function.

```
# First, compute attention weights using the softmax function
queries = self_attention_v2.W_query(inputs) # torch.Size([6, 2])
keys = self_attention_v2.W_key(inputs) # torch.Size([6, 2])
attention_scores = queries @ keys.T # torch.Size([6, 6])
attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

print(attention_weights)
```
* Output from the first step:
```
tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

* **Second**, 2.1 create a mask where the values above the diagonal are zero. This is done by using Pytorch's tril function.

```
# 2.1 create a mask where the values above the diagonal are zero
context_length = attention_scores.shape[0] # 6
# Return a lower triangle
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

* The resulting mask has 0 at the upper triangle and 1 at the diagonal and below.
```
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
```

* **Second**, 2.2 Multiply this mask with the attention weights to zero-out the values above the diagonal:

```
# 2.2 multiply this mask with the attention weights to zero-out the values above the diagonal
masked_simple = attention_weights * mask_simple
print(masked_simple)
```

* The output of the mask shows that the elements above the diagonal are zeroed out!
```
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>)
```

* **Third**, renormalise the attention weights to sum up to 1 again in each row.
* This can be achieved by dividing each element in each row by the sum in each row,

```
# Third, re-normalise attention weights to 1.
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```

* Output: an attiontion weight matrix where the attention weights above the diagonal are zeroed-out and the rows sum to 1.
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)
```

* Note:
```
row_sums = masked_simple.sum(dim=-1, keepdim=True) # torch.Size([6, 1])
> tensor([[0.1921],
        [0.3700],
        [0.5357],
        [0.6775],
        [0.8415],
        [1.0000]], grad_fn=<SumBackward1>)

row_sums = masked_simple.sum(dim=-1, keepdim=False) # torch.Size([6])
> tensor([0.1921, 0.3700, 0.5357, 0.6775, 0.8415, 1.0000],
       grad_fn=<SumBackward1>)
```

> **Information leakage**\
> When we apply a mask and then renormalize the attention weights, it might initially appear that information from future tokens (which we intend to mask) could still influence the current token because their values are part of the softmax calculation.
> However, the key insight is that when we renormalize the attention weights after masking,
> what we’re essentially doing is recalculating the softmax over a smaller subset (since masked positions don’t contribute to the softmax value).\
> The mathematical elegance of softmax is that despite initially including all positions in the denominator, after masking and renormalizing, the effect of the masked positions is nullified - they don’t contribute to the softmax score in any meaningful way.
> \
> In simpler terms, after masking and renormalization, the distribution of attention weights is as if it was calculated only among the unmasked positions to begin with. This ensures there’s no information leakage from future (or otherwise masked) tokens as we intended.

* We can improve the implementation of causal attention further by applying a mathematical property of the softmax function and implement the computation of the masked attention weights more efficiently.

<img width="557" alt="image" src="https://github.com/user-attachments/assets/6a8d0818-c4ea-45c7-8d94-e68dd304dc33" />
 
* Overview steps:
    1) In: Attention scores (unnormalised) ----> Mask with -∞ above diagonal ----> Out: Masked attention scores (unnormalised)
    2) In: Masked attention scores (unnormalised) ----> Apply softmax ----> Out: Masked attention weights (normalised)

* The softmax function converts its inputs into a probability distribution.
* When negative infinity values (-∞) are present in a row, the softmax function treats tthem as a zero probability.
* This is because, mathematically, e^(-∞) = 1 / e^(∞) = 0.

```
# Masked upper triangular with -inf

# Upper triangular part
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
print(masked) 
```

* The output mask is:

```
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
```

* Next, we simply apply the softmax function in which the -∞ will become 0.

```
# Apply the softmax function
attention_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attention_weights)
```

* Output:

```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

* Once we have attention weights, we can compute the context vectors as

```
# Compute the context vector
values = self_attention_v2.W_value(inputs) # torch.Size([6, 2])
context_vector = attention_weights @ values
print(context_vector)
```

* Output:
```
tensor([[-0.0872,  0.0286],
        [-0.0991,  0.0501],
        [-0.0999,  0.0633],
        [-0.0983,  0.0489],
        [-0.0514,  0.1098],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
```

### 3.5.2 Masking additional attention weights with dropout

### 3.5.3 Implementing a compact causal attention class

## 3.6 Extending single-head attention to multi-head attention

## Summary
