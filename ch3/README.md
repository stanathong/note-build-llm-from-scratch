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

## 3.4 Implementing self-attention with trainable weights

## 3.5 Hiding future words with causal attention

## 3.6 Extending single-head attention to multi-head attention

## Summary
