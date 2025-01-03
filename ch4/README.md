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

* Next: To implement the code.


## 4.2 Normalizing activations with layer normalization

## 4.3 Implementing a feed forward network with GELU activations

## 4.4 Adding shortcut connections

## 4.5 Connecting attention and linear layers in a transformer block

## 4.6 Coding the GPT model

## 4.7 Generating text

## Summary
