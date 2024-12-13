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

**START: Hand-on:[code/download-the-verdict.py](code/download-the-verdict.py)**
```
import urllib.request
import re # regular expression

# 1. Download the text file that will be used to train the LLM models
# url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
       "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 2. Load the 'the-verdict.txt' file using Python's standard file reading utilities
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(f'Total number of characters: {len(raw_text)}')
print(raw_text[:5]) # I HAD
print(raw_text[:50]) # I HAD always thought Jack Gisburn rather a cheap g
'''
012345678901
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow 
'''

# 3. Use python's regular expression library to split text on white space char
# import re
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'(\s)', text)
print(splitted)
# The result is a list of individual words, whitespaces and punctuation characters
'''
['I', ' ', 'HAD', ' ', 'always', ' ', 'thought', ' ', 'Jack', ' ', 'Gisburn', 
' ', 'rather', ' ', 'a', ' ', 'cheap', ' ', 'genius--though', ' ', 'a', ' ', 
'good', ' ', 'fellow', ' ', 'enough--so', ' ', 'it', ' ', 'was', ' ', 'no', 
' ', '']
'''

# 4. To avoid the case where some words are still conntected to punctuation characters.
# Modify the regular expression splits on white space \s, comman and period
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'([,.-]|\s)', text)
print(splitted)
'''
['I', ' ', 'HAD', ' ', 'always', ' ', 'thought', ' ', 'Jack', ' ', 'Gisburn', 
' ', 'rather', ' ', 'a', ' ', 'cheap', ' ', 'genius', '-', '', '-', 'though', 
' ', 'a', ' ', 'good', ' ', 'fellow', ' ', 'enough', '-', '', '-', 'so', ' ', 
'it', ' ', 'was', ' ', 'no', ' ', '']
'''

# 5. Remove white space characters
splitted = [item for item in splitted if item.strip()]
print(splitted)
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
'genius', '-', '-', 'though', 'a', 'good', 'fellow', 'enough', '-', '-', 'so', 
'it', 'was', 'no']
'''

# 6. Remove extra special chars
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(splitted)
splitted = [item.strip() for item in splitted if item.strip()]
print(splitted)
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 
'was', 'no']
'''

# 7. Process the whole file
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # 4096
print(preprocessed[:30]) # the first 30 tokens
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 
 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
'''
```
**END: Hand-on**

## 2.3 Converting tokens into token IDs

* Next steps after we have splitted text into words or tokens, we have to convert these tokens from a Python string to an integer representation to **produce the token IDs**.
* These token IDs are just intermediate convertion before converting them into embedding vectors.
* To mapthe generated tokens into token IDs, we have to build a vocabulary first, this vocabulary defines how we map each unique word and special character to a unique integer.

<img width="664" alt="image" src="https://github.com/user-attachments/assets/1d3fa5b9-6b2a-406f-8c4d-4472bdd28f7d" />

```
# 8. Create a list of all unique tokens and sort them alphabetically to determine the vocab size
all_words = sorted(set(preprocessed))
vocab_size = len(all_words) # 1130
print(f'The total number of words in the vocabulary is {vocab_size}')

# 9. Create vocabulary by mapping id to token
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(f'{i} : {item}')
    if i >= 50:
        break

'''
0 : ('!', 0)
1 : ('"', 1)
2 : ("'", 2)
3 : ('(', 3)
4 : (')', 4)
5 : (',', 5)
6 : ('--', 6)
7 : ('.', 7)
8 : (':', 8)
9 : (';', 9)
10 : ('?', 10)
11 : ('A', 11)
12 : ('Ah', 12)
13 : ('Among', 13)
...
'''
```

* From the result above, the dictionary contains individual tokens associated with unique integer labels.
* The next step is to apply this vocabulary to convert new text into token IDs.
* When we want to convert the outputs of an LLM from numbers back into text, we need a way to turn token IDs into text.
* For this, we can create an inverse version of the vocabulary that maps token IDs back to the corresponding text tokens.

<img width="675" alt="image" src="https://github.com/user-attachments/assets/ebc11f93-e659-48b5-8427-54b2052c8ef7" />

* Implement a class SimpleTokenizerV1 that implement a decode and encode method

**START: Hand-on**
```
import re
import urllib.request
import textwrap

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        try:
            ids = [self.str_to_int[token] for token in preprocessed] 
        except KeyError as e:
            raise ValueError(f'Token \'{e.args[0]}\' not found in vocabulary.') from e
        return ids
    
    def decode(self, ids):
        try:
            text = ' '.join([self.int_to_str[i] for i in ids])
        except KeyError as e:
            raise ValueError(f'ID \'{e.args[0]}\' not found in vocabulary.') from e

        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
def main():
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
           "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words) # 1130
    print(f'The total number of words in the vocabulary is {vocab_size}')

    vocab = {token:integer for integer, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab)

    # valid text
    try:
        text =  '''
                "My dear, since I've chucked painting people don't say 
                that stuff about me--they say it about Victor Grindle,"
                '''
        ids = tokenizer.encode(textwrap.dedent(text))
        decoded_text = tokenizer.decode(ids)
        print(decoded_text)
    except ValueError as e:
        print('Error: ', e)

    # invalid text
    try:
        text =  '''
                There is a celebrity hippo family in Thailand. 
                My favourite hippo is Moo Deng.
                '''
        ids = tokenizer.encode(textwrap.dedent(text))
        decoded_text = tokenizer.decode(ids)
        print(decoded_text)
    except ValueError as e:
        print('Error: ', e)
        # Error:  Token 'celebrity' not found in vocabulary.

if __name__ == '__main__':
    main()
```
**END: Hand-on**

