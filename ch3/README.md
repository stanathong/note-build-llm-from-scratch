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
> * The enhanced cntext vector z(2) is an embedding that contains information about x(2) and all other input elements x(1) to x(T).

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

## 3.4 Implementing self-attention with trainable weights

## 3.5 Hiding future words with causal attention

## 3.6 Extending single-head attention to multi-head attention

## Summary
