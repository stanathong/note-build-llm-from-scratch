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

### 3.4.2 Implementing a compact self-attention Python class

## 3.5 Hiding future words with causal attention

## 3.6 Extending single-head attention to multi-head attention

## Summary
