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

## 4.2 Normalizing activations with layer normalization

## 4.3 Implementing a feed forward network with GELU activations

## 4.4 Adding shortcut connections

## 4.5 Connecting attention and linear layers in a transformer block

## 4.6 Coding the GPT model

## 4.7 Generating text

## Summary
