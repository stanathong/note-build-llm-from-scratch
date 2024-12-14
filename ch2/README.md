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

<img width="673" alt="image" src="https://github.com/user-attachments/assets/86597c09-7d17-4b8c-8691-b65ad920e6f9" />

## 2.4 Adding special context tokens

* Refering to the implementation of class SimpleTokenizerV1, we need to modify the tokenizer to
    * Handle unknown words
    * Address the usage and addition of special context tokens that can enhance a models' understanding of context or other relevant information in the text.
    * These special tokens can include markers for unknown words and document boundaries.
 * **Now, we will modify the vocab and tokenizer, SimpleTokenizerV1 to SimpleTokenizerV2, to support two new tokens: `<|unk|>` and `<|endoftext|>`**.

<img width="645" alt="image" src="https://github.com/user-attachments/assets/c8d8f714-15e0-40e8-ac84-4d0fae80c2dc" />

**Cases:**
* **`<|unk|>`** token is used if enchounter a word that is not part of the vocab.
* **`<|endoftext|>`** token is added between unrelated text when training GPT-like LLMs on multiple independent documents or books.
    * It is common to insert a token before each document that follows a previous text source.
    * This helps the LLM understand that although these text sources are concatenated for training, the are in fact unrelated. 

<img width="650" alt="image" src="https://github.com/user-attachments/assets/2a68e1b5-1b95-4ea2-96b4-0edede766ec4" />

**START: Hand-on:[code/build-vocab-from-the-verdict.py](code/build-vocab-from-the-verdict.py)**

```
import urllib.request
import re # regular expression

# 1. Download the text file that will be used to train the LLM models
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
       "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_name = "the-verdict.txt"
urllib.request.urlretrieve(url, file_name)

# 2. Load the 'the-verdict.txt' file using Python's standard file reading utilities
with open(file_name, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f'Total number of characters: {len(raw_text)}')
# Total number of characters: 20479
print(raw_text[:10]) # I HAD alwa
print(raw_text[-10:]) # d of art."

# 3. Preprocess the file by using python's regular expression lib 
#    - to split text on while space char \s
#    - to split words connected to special chars e.g. punctuation, comma, etc
#    - to split text using ,.:;?_!"()\', --, space (\s)
preprocessed = re.split(r'[,.:;?_!"()\']|--|\s', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f'Total number of all tokens: {len(preprocessed)}') #  3788
print(preprocessed[:10]) # The first 10 tokens

# 4. Create a list of all unique tokens and sort them alphabetically
all_tokens = sorted(set(preprocessed))
vocab_size = len(all_tokens)
print(f'Total number of unique tokens: {vocab_size}') # 1118

# 5. Extend the set of tokens to include <|unk|> and <|endoftext|>
all_tokens.extend(['<|unk|>', '<|endoftext|>'])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(f'Total number of vocab tokens: {len(vocab)}') # 1120
# Print the last 5 items
for id, item in list(vocab.items())[-5:]:
    print(f'{id} : {item}')

'''
Output:
-------
Total number of characters: 20479
I HAD alwa
d of art."
Total number of all tokens: 3788
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius']
Total number of unique tokens: 1118
Total number of vocab tokens: 1120
younger : 1115
your : 1116
yourself : 1117
<|unk|> : 1118
<|endoftext|> : 1119
'''
```

**END: Hand-on**

* Implement a class SimpleTokenizerV2 that implement a decode and encode method.
* See the implementation in [code/simple-tokenizer-v2.py](code/simple-tokenizer-v2.py).

**START: Hand-on**

```
import re
import urllib.request
import textwrap

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Replaces unknown words by <|unk|> tokens
        preprocessed = [item if item in self.str_to_int 
                        else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed] 
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

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(['<|endoftext|>', '<|unk|>'])
    vocab_size = len(all_tokens) # 1132
    print(f'The total number of words in the vocabulary is {vocab_size}')

    vocab = {token:integer for integer, token in enumerate(all_tokens)}
    tokenizer = SimpleTokenizerV2(vocab)

    # invalid text which is now handled in v2
    try:
        text1 = 'My favourite hippo is Moo Deng.'
        text2 = 'Structure from Motion and SLAM are both used to map the scene.'
        text = ' <|endoftext|> '.join((text1, text2))
        ids = tokenizer.encode(textwrap.dedent(text))
        decoded_text = tokenizer.decode(ids)
        print(text)
        print(ids)
        print(decoded_text)
    except ValueError as e:
        print('Error: ', e)
        # Error:  Token 'celebrity' not found in vocabulary.

if __name__ == '__main__':
    main()
```

**Output:**
```
My favourite hippo is Moo Deng. <|endoftext|> Structure from Motion and SLAM are both used to map the scene.
[68, 1131, 1131, 584, 1131, 1131, 7, 1130, 1131, 477, 1131, 157, 1131, 169, 1131, 1057, 1016, 1131, 988, 1131, 7]
My <|unk|> <|unk|> is <|unk|> <|unk|>. <|endoftext|> <|unk|> from <|unk|> and <|unk|> are <|unk|> used to <|unk|> the <|unk|>.
```
**END: Hand-on**

### Practical Implementation

* Depending on the LLM, some researchers also consider additional special tokens such as the following:
* [BOS] - beginning of sequence : This token marks the start of a text. It tells LLM where a piece of text begins.
* [EOS] - end of sequence : This token is positioned at the end of a text. 
* [PAD] - padding : When training LLM with batch sizes larger than one, the batch might contain texts of varying lengths. To ensure all texts have the same length, the shorter texts are extended or padded using the [PAD] token upto the length of the longest text in the batch.
* However, the tokenizer used for GPT models does not need any of these tokens - as <|endoftext|> token can be used for simplicity because it can be used for [EOS]  and for [PAD].
* However when trianing on batched inputs, we typically use a **mask**, meaning we don't attend to padded tokens.
* Also, the tokenizer used for GPT models also doesn't use an <|unk|> token for out-of-vocabulary words.
* Instead, GPT models use a `byte pair encoding` tokenizer, which breaks words down into subword units.


## 2.5 Byte pair encoding (BPE)

* A more sophisticated tokenization scheme based on a concept called *byte pair encoding (BPE)*.
* The BPE tokenizer was used to train LLMs such as GPT-2, GPT-3 and the original model used in ChatGPT.
* GPT-2 used BytePair encoding (BPE) as its tokenizer.
* BPE allows the model to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to **handle out-of-vocabulary words.**
    * For instance, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges
* The original BPE tokenizer can be found here: [https://github.com/openai/gpt-2/blob/master/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
    * However, we are using the BPE tokenizer from OpenAI's open-source [tiktoken](https://github.com/openai/tiktoken) library, which implements its core algorithms in Rust to improve computational performance

#### Setup

1. Install the tiktoken package
```
pip install tiktoken
```
2. Check the version that is just installed.

```
>>> from importlib.metadata import version
>>> import tiktoken
>>> print('tiktoken version: ', version('tiktoken'))
```

**Output:**

```
tiktoken version:  0.8.0
```

#### Tiktoken

```
from importlib.metadata import version
import tiktoken

# Check version of tiktoken
print('tiktoken version: ', version('tiktoken')) # 0.8.0

# Once installed, we can instantiate the BPE tokenizer from tiktoken as follows:
tokenizer = tiktoken.get_encoding('gpt2') 

text = (
    'My favourite hippo is Moo Deng or MooDeng. <|endoftext|> Structure from Motion '
    'and SLAM are both used to map the scene.'
)

integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print('Type of integers: ', type(integers)) # <class 'list'>
print(integers)

# Convert the token id back into text using decode method
strings = tokenizer.decode(integers)
print(strings)

'''
[3666, 12507, 18568, 78, 318, 337, 2238, 41985, 393, 337, 2238, 35, 1516, 13, 220, 50256, 32522, 422, 20843, 290, 12419, 2390, 389, 1111, 973, 284, 3975, 262, 3715, 13]
My favourite hippo is Moo Deng or MooDeng. <|endoftext|> Structure from Motion and SLAM are both used to map the scene.
'''
```

* The <|endoftext|> token is assigned to relatively large token ID = 50256.
* In fact, the BPE tokenizer which was used to train models such as GPT-2, GPT3 and the original model used in ChatGPT has a total vocab size of 50257, with <|endoftext|> being assigned the largest token ID.
* The BPE tokenizer encodes and decodes unknown words such as MooDeng.
* **The algorithm underlying BPE breaks down words that aren't in its predefined vocab into smaller subword units or even individual chars, enabling it to handle out-of-vocabularly words.**. As a result, if the tokenizer encounters an unfamiliar word during tokenization, it can represent it as a sequence of subword tokens or characters.

<img width="648" alt="image" src="https://github.com/user-attachments/assets/85518a44-d782-49b9-a63e-fbf5c2404425" />


> The ability to break down unknown words into individual chars ensures that
> the tokenizer and, consequently, the LLM that is trained with it can process
> any text, even if it contains words that were not present in its training data.

#### Test BPE tokenizer on unknown word: Akwirw ier

```
text = 'Akwirw ier'
integers = tokenizer.encode(text)
strings = tokenizer.decode(integers)
print(text) # Akwirw ier
print(integers) # [33901, 86, 343, 86, 220, 959]
print(strings) # Akwirw ier
```





