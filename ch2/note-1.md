# Chapter 2: Working with text data

## Introduction

* The models used in ChatGPT and other GPT-like LLMs are decoder-only LLMs based on the transformer architecture.
* During the pretraining stage, LLMs process text one word at a time. Training LLMs with million to billions of parameters using a next word prediction task.
* Preparing input text for training LLMs involves splitting text into individual word and subword tokens, which can then be encoded into vector representations for the LLMs.

<img width="760" alt="image" src="https://github.com/user-attachments/assets/7907482e-0333-4bac-9704-76fca854ec06" />

## 2.1 Understanding word embeddings

* Deep neural network models, LLM included, cannot process raw text directly it isn't compatible with the maths operations used to implement and train NN.
* We need a way to represent words as continuous-valued vectors.
* The concept of converting data into a vector format is often referred to as **embedding**.
* We use an embedding model to transform video, audio, and text into vector representation using a specific NN layer or another pretrained NN models.
* Different data formats require distinct embedding models.

 <img width="662" alt="image" src="https://github.com/user-attachments/assets/dfcd98b4-3837-456c-99f4-12c252804c6e" />

 * At its core, an **embedding is a mapping from discrete objects** such as words, images or even entire documents, **to points in a continous vector space**.
 * **The primary purpose of embeddings is to convert non numeric data into numeric data that neural networks can process.**
 * There are more than just word embeddings such as embeddings for sentences, paragraph or the whole documents. The latter two are popular choices for retrieval-augmented generation.
 * Retrieval augmented generation combines generation (like producing text) with retrieval (like searching an external knowledge base) to pull relevant information when generating text.
 * One of the most popular algorithms/frameworks for word embeddings is Word2Vec. Word2Vec trained neural network architecture to generate word embeddings by predicting the context of a word given the target word or vice versa. The main idea behand Word2Vec is that words that appear in similar contexts tend to have similar meanings. Consequently, when projected into 2D word embeddings for visualisation purposes, similar terms are clustered together.
 * Word embeddings can have varying dimensions. A higher dimensionality might capture more nuanced relationships but at the cost of computational efficiency. 

<img width="589" alt="image" src="https://github.com/user-attachments/assets/4d2d02e9-57d2-4109-a279-27aae386d77f" />

* While we can use prerained models such as Word2Vec to generate embeddings for ML models, LLMs commonly produce their own embeddings that are part of the input layer and are updated during training.
* The advantage of optimising embedding as part of LLM training instead of using Word2Vec is that the embeddings are optimised to the specific task and data at hand.
* When working with LLMs, we typically use embeddings with a much higher dimensionality.
* The general steps for preparing embeddings used in an LLM: **splitting text into words --> converting words into tokens --> converting token into embedding vectors**.

## 2.2 Tokenizing Text

* Tokenizing text involves how we split input text into individual tokens, a required preprocessing stop for creating embeddings for an LLM.
* These tokens are either individual words or special characters including punctuation characters.

<img width="632" alt="image" src="https://github.com/user-attachments/assets/1a04d546-07e3-4765-b6f3-cbfe39fe831b" />


**Hand-on**






