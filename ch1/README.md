# Chapter 1 - Understanding large language models

## Introduction

* Large language models (LLM) are deep neural network models.
* LLM is a new era for NLP (natural language processing).
* Traditional methods can perform tasks such as email spam classification which can be captured with handcrafted rules or simpler models.
* The tranditional methods underperformed in language tasks that require complex understanding and generating text e.g.
    * parsing detailed instruction
    * conducting contextual analysis
    * creating coherent and contextually appropriate text
* LLMs are capable to **understand, generate, and interpret human language**, in which they can process and generate text in ways that appear coherent and contextually relevant.
* LLMs are trained on vast quantities of text data This large-scale training allows LLMs to capture **deeper contextual information and subtleties of human language**.
* LLMs have improved performance in wide range of NLP tasks such as text translation, sentiment analysis, question answering etc.
* **Another important distinction between contemporary LLMs and earlier NLP models is that earlier NLP models were typically designed for specific tasks such as text categorization, language translation etc. (NLP models exelled in their narrow applications) LLMs demonstrate a broader proficiency across a wide range of NLP tasks**
* The success behind LLMs can be attributed to (1) transformer architecture, (2) the vast amount of data on which LLMs are trained.

## 1.1 What is an LLM?

<img width="741" alt="image" src="https://github.com/user-attachments/assets/58ffed13-5509-42df-b493-2b75745926b6" />

* An LLM is a NN designed to understand, generate and respond to human-like text.
* The term *large* in *large language model* refers to both the model's size in term of parameters and the immense dataset on which it's trained.
* LLM models often have tens or even hundreds of billions of parameters, which are adjustable weights in the network that are optimised during **training to predict the next word in the sequence**.
* LLMs utilise an architecture called the *transformer*, which allows them to pay selective attention to different parts of the input when making predictions.
* Based on the fact that **LLMs can generate text**, LLMs are also referred to as Generative AI or GenAI.

## 1.2 Applications of LLMs

* Due to their advanced capabilities to parse and understand unstructured text data, LLMs have a broad range of applciations across various domains e.g.
    * machine translation
    * generation of novel texts
    * sentiment analysis
    * text summarisation
    * content creation
    * chatbot and virtual assistants such as ChatGPT or Gemini which can answer user queries and augment traditional search engines. 
* In short, **LLMs are invaluable for automating almost any task that involves parsing and generating text**.

## 1.3 Stages of building an using LLMs

* Regarding model performance, custom-built LLMs, LLMs that are tailored for specific tasks or domains, can outperform general-purpose LLMs like ChatGPT.
* Example of custom-built LLMs include BloombergGPT (specialised for finance) and LLMs tailored for medical question answering.
* Using custom-built LLMs offers several advantages, particularly **privacy**.
* Developing smaller custom LLMs enables deployment directly on customer devices such as laptops and smartphones. This decreases latency and reduces server-related costs.
* The general process of crating an LLM includes pre-training and fine-tuning.
    * **pre-trianing**: the initial phase where an LLM model is trained on a large, diverse dataset to develop a broad understanding of language. This pretrained model serves as **a foundation model**.
    * **fine-tuning**: the process where the model is trained on a narrower, specific dataset that is more specific to particular tasks or domains.  

<img width="690" alt="image" src="https://github.com/user-attachments/assets/c3b2261a-28c6-4e8f-b3a4-dab17ba4bf6e" />

* The first step in creating an LLM is to train it on a large raw text data i.e. regular text without any labeling information. The reason that LLMs do not required labeling information because in the pretraining, LLMs use self-supervised learning where the model generates its own labels from the input data. A pre-trained LLM is obtained from training it to predict the next word in text. This pre-training stage creates a trained LLM which is often called **a base or foundation model**. They are capable of text completion, (limited) few-shot capacities which means it can learn to perform new tasks based on only a few examples.
* The second phase is fine tuning. After obtaining a pre-trained LLM, we can further train the LLM on labeled data.

```
The two most popular categories of fine-tuning LLMs are (1) instruction fine-tuning
and (2) classification fine-tuning.

1. Instruction fine-tuning
* The labeled dataset consits of instruction and answer pairs, such as a query to
translate a text accompanied by the correctly translated text.

2. Classification fine-tuning
* The labelled dataset consists of texts and associated class labels. For example,
emails associated with `spam` and `not spam` labels.
```

## 1.4 Introducing the transformer architecture

* Most modern LLMs rely on the transformer architecture.
* A simplified version of the transformer architecture is depicted below:

<img width="731" alt="image" src="https://github.com/user-attachments/assets/479d8953-42ee-4bcc-a17c-6332fd618fec" />

* The transformer architecture consists of 2 submodules: an encoder and a decoder.
    * The encoder module processes the input text and **encodes it into a series of numerical representations or vectors** that capture the contextual information of the input.
    * The decoder module takes the encoded vectors and generates the output text.  

* Both encoder and decoder consists of many layers connected by a self-attention mechanism.
* A key component of transformers and LLMs is the self-attention mechanism, which allows the model to weigh the importance of different words or tokens in a sequence relative to each other. This mechanism enables the model to capture long-range dependencies and contexual relationships within the input data, enhancing its ability to generate coherent and contexually relevant output. 
* Variants of the transformer architecture: BERT (bidirectional encoder representations from transformer) and GPT (generative pretrained transformers).
* **BERT** is built upon the original transformer's encoder submodule. It specialises in **masked word prediction** where the model predicts masked or hidden words in a given sentence --> giving BERT strengths in text classification tasks including sentiment prediction and document catorization.
* **GPT** focuses on the decoder portion of the original transformer and is designed for tasks that require **generating texts**. GPT models are primarily designed and trained to perform text completion tasks and also show remarkable versatility in their capabilities. These models are adept at executing both zero-shot and few-shot learning tasks.
    * Zero shot learning is the ability of the model to generalise to completely unseen tasks without any prior specific examples.
    * Few-shot learning involves learning from a minimum number of examples the user provides as input. 

* **Note: BERT is built upon the encoder while GPT is built upon the decoder.**
<img width="727" alt="image" src="https://github.com/user-attachments/assets/deb9c8db-2603-4bab-8da2-a9036d766ebc" />

* **Zero-shot learning vs Few-shot learning**

<img width="755" alt="image" src="https://github.com/user-attachments/assets/d739fd5c-ca1b-4081-bebd-bbbdb07ceb17" />

## 1.5 Utilizing large datasets

* The large training datasets for popular GPT-like and BERT-like models represent diverse and comprehensive text corpos encompasing billions of words.
* Below is the summary of the dataset used for pre-training GPT-3 which served as the base model for the first version of ChatGPT.

<img width="679" alt="image" src="https://github.com/user-attachments/assets/b2c14683-d7f7-4461-b198-525c715ed3dc" />

* The main takeaway is that the scale and diversity of this training dataset allow these models to perform well on diverse taks, including language syntax, semantics and context. 

## 1.6 A closer look at the GPT architecture

* **GPT models are pretrained on a relatively simple next-word prediction task.**

<img width="657" alt="image" src="https://github.com/user-attachments/assets/d2b96b9a-a6a8-48a5-8765-52f4e16fe327" />

* The next-word prediction task is a form of self-supervised learning, which is a form of self-labeling.
* We don't need to collect labels for the training data explicitly but can use the structure of the data itself: we can use the next word in a sentence or document as the label that the model is supposed to predict.
* Therefore, it is possible to use massive unlabeled text datasets to train LLMs.
* Compared to the original trasformer architecture, the general GPT architecture is relatively small. It's just the decoder part without encoder.
* Since decoder-style models like GPT generate text by predicting text one word at a time, they are considered a type of **autoregressive** model.
* Autoregressive models incorporate their previous outputs as inputs for future predictions.
* In GPT, each new word is chosen based on the sequence that precedes it, which improves the coherence of the resulting text.
* Architectures such as GPT-3 are signficantly larger than the original transformer. For example, the original transformer repeated the encoder and decoder blocks six times while GPT-3 has 96 trnasformer layers and 175 billion parameters.

<img width="745" alt="image" src="https://github.com/user-attachments/assets/24411627-060b-4dbd-b0da-bba69bfad08b" />

* The original transformer model consisting of encoder and decoder blocks. GPT models are, despite larger, simpler as they consist only of decoder aimed at next word prediction. 

## 1.7 Building a large language model

<img width="754" alt="image" src="https://github.com/user-attachments/assets/83b99d65-5ffa-4ba1-b9dd-8cb3e85c6757" />

## Summary

* Modern LLMs are trained in 2 main steps:
    * pre-training: they are pretrained on a large corpus of unlabeled text by using the prediction of the next word in a sentence as a label.
    * finetuning: they are fine-tuned on a smaller, labeled target dataset to follow instructions or perform classification tasks.
* LLMs are based on the transformer architecture. The key idea of the transformer architecture is an attention mechanism.
* The original transformer architecture consists of an encoder for parsing text and a decoder for generating text.
* LLMs for generating text and following instructions e.g. GPT-3 and ChatGPT only implement decoder modules.
* The general pretraining task for GPT-like models is to predict the next word in the sentence.
* Once an LLM is pretrained, the resulting foundation model can be finetuned more efficiently for various downstream tasks. 







 
