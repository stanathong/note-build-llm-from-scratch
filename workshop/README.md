# Building LLMs from Ground Up

**Resource:** 
* [Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://www.youtube.com/watch?v=quh7z1q7-uc)
* https://github.com/rasbt/LLM-workshop-2024
* https://github.com/Lightning-AI/litgpt
* https://lightning.ai/lightning-ai/studios/llms-from-the-ground-up-workshop?section=featured

There are a few way to use LLMs:

1. Via public / proprietary services. For example, Chat GPT.
2. Run a (custom) LLM **locally** e.g. Llama, Microsoft Phi 3.
3. Deploy an LLM API end point / server. Use API end point to access the LLM that we deploy on the server.

## Developing an LLM

<img width="977" alt="image" src="https://github.com/user-attachments/assets/c81b4b56-246e-4d6c-a98c-ee52788d27bf" />

* We will focus on these topics.
<img width="1011" alt="image" src="https://github.com/user-attachments/assets/2984a505-85d3-4e25-83f3-d7c825ebaa13" />

<img width="1003" alt="image" src="https://github.com/user-attachments/assets/b1b86ca3-1457-4ea7-9d7d-0d2f617f76f9" />

* Two key repos:
    * https://github.com/rasbt/LLM-workshop-2024
    * https://github.com/Lightning-AI/litgpt

## Setup

* This is how to set up locally: https://github.com/rasbt/LLM-workshop-2024/tree/main/setup

1. Clone the repo:

```
cd ~/Dev/Learning/LLM-suite
git clone git@github.com:rasbt/LLM-workshop-2024.git
```

2. Install [miniforge](https://github.com/conda-forge/miniforge), see https://github.com/rasbt/LLM-workshop-2024/tree/main/setup#1-download-and-install-miniforge. Since I already have Anaconda installed on my machine, I will skip this.

3. To speed up Conda, you can use the following setting, which switches to a more efficient Rust reimplementation for solving dependencies: (I skipped this.)

```
conda config --set solver libmamba
```

4. Create new environment.

```
conda create -n LLMs python=3.10
conda activate LLMs
```

5. Install the required packages.

```
pip install -r requirements.txt
```

The requirements.txt file contains the following packages:

```
torch >= 2.0.1 
jupyterlab >= 4.0
tiktoken >= 0.5.1
matplotlib >= 3.7.1
numpy >= 1.24.3
tensorflow >= 2.15.0
tqdm >= 4.66.1
numpy >= 1.25, < 2.0
pandas >= 2.2.1
psutil >= 5.9.5
litgpt[all] >= 0.4.1
```

# Section 2 - Data Preparation

* We'll follow the jupyter notebook for Section 2. I open it using jupyter-lab.
* In this section, we'll cover the process of converting intput text -> tokenized text -> token ids. Typically, after this step, we will perform token embeddings, but it won't be included in this workshop. 

<img width="692" alt="image" src="https://github.com/user-attachments/assets/f24a6e4c-1917-4f7e-84e5-01496bb31377" />

```
from importlib.metadata import version

print("torch version:", version("torch")) # torch version: 2.4.1
print("tiktoken version:", version("tiktoken")) # tiktoken version: 0.8.0
```

```
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

Output:
```
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
```

<img width="686" alt="image" src="https://github.com/user-attachments/assets/259544fc-e7a8-4d3f-bf4f-554c570d9b22" />

* We perform tokenizing text by simply using regular expression to split text by extra characters e.g. punctuations.

```
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item]
print(preprocessed[:38])
```

* Output:
```
['I', ' ', 'HAD', ' ', 'always', ' ', 'thought', ' ', 'Jack', ' ', 'Gisburn', ' ', 'rather', ' ', 'a', ' ', 'cheap', ' ', 'genius', '--', 'though', ' ', 'a', ' ', 'good', ' ', 'fellow', ' ', 'enough', '--', 'so', ' ', 'it', ' ', 'was', ' ', 'no', ' ']
```

```
print("Number of tokens:", len(preprocessed)) # 8405
print('Number of uniqued tokens:', len(set(preprocessed))) # 1132
sorted(set(preprocessed))[-20:] # To show list of the last 20 unique tokens
```

* Output:
```
['wits',
 'woman',
 'women',
 'won',
 'wonder',
 'wondered',
 'word',
 'work',
 'working',
 'worth',
 'would',
 'wouldn',
 'year',
 'years',
 'yellow',
 'yet',
 'you',
 'younger',
 'your',
 'yourself']
```

## Convert tokens into token IDs
* This is done by using vocabulary e.g. dictionary which is just a unique mapping from words to integer.

<img width="695" alt="image" src="https://github.com/user-attachments/assets/8d507b02-a3d8-45ea-8324-577b5e56d42c" />

```
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size) # 1132
```

```
for i in enumerate(all_words):
    print(i)
```

* Examples of words
```
(0, '\n')
(1, ' ')
(2, '!')
(3, '"')
(4, "'")
...
```

* Build a vocabulary and print out the first 20 tokens
```
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if (i >= 20):
        break
```

<img width="693" alt="image" src="https://github.com/user-attachments/assets/61e59842-5965-4ebc-87ba-cc15dc5c857f" />

```
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

* Python note:
```
# This will only print keys
for i in vocab:
    print(i)
'''
!
"
'
(
)
,
'''

for token,i in vocab.items():
    print(f'{token} : {i}')
'''
 : 0
  : 1
! : 2
" : 3
' : 4
'''
```

* Python note for `re.split(r'([,.?_!"()\']|--|\s)', text)`
```
In the regular expression re.split(r'([,.?_!"()\']|--|\s)', text), the \s is a shorthand character class that matches *any whitespace character*. Specifically, it matches:

Spaces (' '), Tabs ('\t'), Newlines ('\n'), Carriage returns ('\r'), Vertical tabs ('\v'), Form feeds ('\f')
```

* Create a tokenizer class

```
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

* Test
```
tokenizer = SimpleTokenizerV1(vocab)
text = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow"
ids = tokenizer.encode(text)
print(ids)
'''
[55, 46, 151, 1005, 59, 40, 820, 117, 258, 488, 8, 1004, 117, 502, 437]
'''
text = tokenizer.decode(ids)
print(text)
'''
I HAD always thought Jack Gisburn rather a cheap genius -- though a good fellow
'''
```

## 2.3 BytePair encoding

* We have a more sophisticate tokenizer which is trained to tokenize text.
* GPT typically use <|endoftext|> for text deliminator.
* GPT-2 used BytePair encoding (BPE) as its tokenizer
* We are using the BPE tokenizer from OpenAI's open-source tiktoken library, which implements its core algorithms in Rust to improve computational performance
* (Based on an analysis [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/02_bonus_bytepair-encoder/compare-bpe-tiktoken.ipynb), I found that tiktoken is approx. 3x faster than the original tokenizer and 6x faster than an equivalent tokenizer in Hugging Face.

```
# pip install tiktoken
import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken")) # 0.8.0
```

```
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
'''
[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
'''
```

```
strings = tokenizer.decode(integers)
print(strings)
```

```
Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
```

* My test

```
encoded = tokenizer.encode('Moodeng deng oohlala')
print(encoded) # [44, 702, 1516, 288, 1516, 267, 48988, 6081]

decoded = tokenizer.decode(encoded)
print(decoded) # Moodeng deng oohlala

tokenizer.decode([44]) # 'M'

for id in range(len(encoded)):
    print( id,':', tokenizer.decode([encoded[id]]) )
'''
0 : M
1 : ood
2 : eng
3 :  d
4 : eng
5 :  o
6 : ohl
7 : ala
'''
```
* BPE tokenizers break down unknown words into subwords and individual characters:

<img width="700" alt="image" src="https://github.com/user-attachments/assets/bb1260e4-b233-4fb8-8fc6-159859eac5f6" />

## 2.4 Data sampling with a sliding window

* LLM see the blue text and predict the next word (red)
* We have context vector length, we feed in input and the target will shift the input by one

<img width="718" alt="image" src="https://github.com/user-attachments/assets/59ccec98-ce61-48e3-8663-7d80cb58b2ee" />

<img width="714" alt="image" src="https://github.com/user-attachments/assets/72516c6f-953d-415f-9413-3c53d18c5eb3" />

<img width="677" alt="image" src="https://github.com/user-attachments/assets/33c6f08e-5874-4d84-a33b-10c2b5f56681" />

* The target will always shift by 1. 

```
from supplementary import create_dataloader_v1


dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```
* Context length is set to 4 and we can see that there is no overlap between each row.
* The target is just shift the input by 1.
* We have batch size of 8.
```
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
```
* supplement.py

```
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
```

# Section 3. Coding an LLM architecture

<img width="673" alt="image" src="https://github.com/user-attachments/assets/9e005f95-ed5c-46ac-966b-fc93cc29d054" />

* The way LLM work is we have to pass the initial text and the LLM will output the next word.
* In this notebook, we consider embedding and model sizes akin to a small GPT-2 model.
* We'll specifically code the architecture of the smallest GPT-2 model (124 million parameters), as outlined in Radford et al.'s Language Models are Unsupervised Multitask Learners (note that the initial report lists it as 117M parameters, but this was later corrected in the model weight repository)

<img width="996" alt="image" src="https://github.com/user-attachments/assets/9815b9db-493c-44f3-b7c8-4bd2b5d4746f" />

* The next notebook will show how to load pretrained weights into our implementation, which will be compatible with model sizes of 345, 762, and 1542 million parameters
* Models like Llama and others are very similar to this model, since they are all based on the same core concepts

<img width="989" alt="image" src="https://github.com/user-attachments/assets/eb572752-abe4-4be9-9152-f515e9a43cfe" />

* Configuration details for the 124 million parameter GPT-2 model (GPT-2 "small") include:

```
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

## 3.2 Coding the GPT model

<img width="745" alt="image" src="https://github.com/user-attachments/assets/18e56641-8740-46ba-9789-8a6de30a77e8" />

* The text will get tokenize and go through the output layer, which we will get a big tensor.
* 4 is the context length (number of tokens) and 50257 columns which is corresponding to the number of vocabulary. 

```
  vocab1  vocab2  vocab3...  # of vocabulary

  item0   item1   item2 .... item as the column for the output vector.
```

```
import torch.nn as nn
from supplementary import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

```
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
```

```
tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])
```

```
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape) # torch.Size([2, 4, 50257])
print(out) # 50257 corresponding to the vocab
```

```
Input batch:
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

# We have 2 batch, 4 tokens, each tokens has 50257 embedding vector
Output shape: torch.Size([2, 4, 50257])

out[0][0].shape # torch.Size([50257])
```

## 3.4 Generating text

* LLM is about generating the next word.
* Each row, its length is the number of vocab words.
* Therefore, the last token will provide the potential word which is the word with highet probability.
<img width="733" alt="image" src="https://github.com/user-attachments/assets/b8bb8199-b4d4-4c2d-923b-2e866d43083f" />


* In this function, what it does is to generate the next word and then append it to the original token and then process contininue.
```
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  # getting the last low, for all column

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```

* This is how it work:

<img width="745" alt="image" src="https://github.com/user-attachments/assets/3be2f768-87e8-47eb-a56d-f8e75e9a8fb3" />

* Exercise: Generate some text
    1. Use the tokenizer.encode method to prepare some input text
    2. Then, convert this text into a pytprch tensor via (torch.tensor)
    3. Add a batch dimension via .unsqueeze(0)
    4. Use the generate_text_simple function to have the GPT generate some text based on your prepared input text
    5. The output from step 4 will be token IDs, convert them back into text via the tokenizer.decode method

```
model.eval();  # disable dropout

text = 'After a year of working hard'
enc_text = torch.tensor(tokenizer.encode(text))
out_ids = generate_text_simple(model=model, idx=enc_text.unsqueeze(0), max_new_tokens=20, context_size=1024)
decode_text = tokenizer.decode(out_ids.squeeze().tolist())

decode_text
'''
'After a year of working hard66666666 previouslypack MB portion312atsHealthigenttarget volunteered total Xen attributeNJ 276 Machina", intertwinedRepl'
'''
# It doesn't make sense because we haven't trained our model yet
```

* Detailed step
```
text = 'After a year of working hard'
enc_text = torch.tensor(tokenizer.encode(text))
enc_text # tensor([3260,  257,  614,  286, 1762, 1327])
enc_text.shape # torch.Size([6])
enc_text.ndim # 1
```

```
# We need to add batch dimension
print(enc_text.unsqueeze(0).shape) # torch.Size([1, 6])
enc_text.unsqueeze(0).ndim # 2

batch_input = enc_text.unsqueeze(0) # torch.Size([1, 6])
batch_input
# tensor([[3260,  257,  614,  286, 1762, 1327]])

out_ids = generate_text_simple(model=model, idx=batch_input, max_new_tokens=20, context_size=1024)
tokenizer.decode(out_ids.squeeze().tolist())
'''
'After a year of working hard66666666 previouslypack MB portion312atsHealthigenttarget volunteered total Xen attributeNJ 276 Machina", intertwinedRepl'
'''
```

# Section 4. Pretraining LLMs

```
from importlib.metadata import version

pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")
```

```
matplotlib version: 3.10.0
numpy version: 1.26.4
tiktoken version: 0.8.0
torch version: 2.4.1
```

<img width="685" alt="image" src="https://github.com/user-attachments/assets/a1f7b277-025c-4320-9d3b-2842d3c7b37a" />

```
import torch
from supplementary import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference
```

<img width="844" alt="image" src="https://github.com/user-attachments/assets/a9a52f5a-26af-479d-8aee-a09b7264c8f2" />

```
import tiktoken
from supplementary import generate_text_simple

# Convert text to a list of token ids and create tensor from it.
# Make sure we add batch dimension
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

# Remove the batch dimension from token id and decode.
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
```

```
start_context = "Looking for opportunity"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

```
Output text:
 Looking for opportunityroleumestrocriminal reminderCarl EconomicNS fourteensim issuer
```

* As we can see above, the model does not produce good text because it has not been trained yet
* How do we measure or capture what "good text" is, in a numeric form, to track it during training?
* The next subsection introduces metrics to calculate a loss metric for the generated outputs that we can use to measure the training progress

## 4.2 Preparing the dataset loaders

* We use a relatively small dataset for training the LLM (in fact, only one short story)
* The training finishes relatively fast (minutes instead of weeks), which is good for educational purposes
* For example, Llama 2 7B required 184,320 GPU hours on A100 GPUs to be trained on 2 trillion tokens
* Below, we use the same dataset we used in the data preparation notebook earlier

```
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()
```

```
# First 100 characters
print(text_data[:99])
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 

# Last 100 characters
print(text_data[-99:])
# it for me! The Strouds stand alone, and happen once--but there's no exterminating our kind of art."
```

```
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters) # Characters: 20479
print("Tokens:", total_tokens) # Tokens: 5145
```

<img width="721" alt="image" src="https://github.com/user-attachments/assets/f1670973-82de-4032-abc0-49b65ee0f559" />

```
from supplementary import create_dataloader_v1


# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
```

```
print("Train loader:")
# x: input, y: target
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
```

```
Train loader:
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])

Validation loader:
torch.Size([2, 256]) torch.Size([2, 256])
```

* An optional check that the data was loaded correctly:
* This is done by counting number of elements in a tensor using numel().

```
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens) # Training tokens: 4608
print("Validation tokens:", val_tokens) # Validation tokens: 512
print("All tokens:", train_tokens + val_tokens) # All tokens: 5120
```

* Next, let's calculate the initial loss before we start training.


```
from supplementary import calc_loss_loader

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss) # 10.987582842508951
print("Validation loss:", val_loss) # 10.98110580444336
```

* Supplementary code
```
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```
* Note for logits.flatten(0, 1)
    * If logits has a shape like (batch_size, seq_len, num_classes), this flattens the first two dimensions into one (i.e., (batch_size * seq_len, num_classes)).
* `torch.nn.functional.cross_entropy` expects predictions (logits): Unnormalized scores and targets (target_batch): Ground-truth labels.
    * Automatically applies log_softmax to the predictions and computes the negative log likelihood.

<img width="552" alt="image" src="https://github.com/user-attachments/assets/6fec33b3-02db-4272-90d2-bea307ae25ef" />
<img width="540" alt="image" src="https://github.com/user-attachments/assets/6b7d7b6c-1e9b-42d5-b28d-fb57a42f8973" />

<img width="892" alt="image" src="https://github.com/user-attachments/assets/918a002a-d7ec-4a70-b4ee-bb3f92ea18c1" />

## 4.2 Training an LLM

<img width="696" alt="image" src="https://github.com/user-attachments/assets/128c54b6-6a9d-4f04-943a-c182d35fcb44" />

```
from supplementary import (
    calc_loss_batch,
    evaluate_model,
    generate_and_print_sample
)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

```
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

```
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 5#10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
```

```
Ep 1 (Step 000000): Train loss 9.817, Val loss 9.924
Ep 1 (Step 000005): Train loss 8.066, Val loss 8.332
Every effort moves you,,,,,,,,,,,,.                                     
Ep 2 (Step 000010): Train loss 6.619, Val loss 7.042
Ep 2 (Step 000015): Train loss 6.046, Val loss 6.596
Every effort moves you, and,, and, and,,,,, and, and,,,,,,,,,,, and,, the,, the, and,, and,,, the, and,,,,,,
Ep 3 (Step 000020): Train loss 5.524, Val loss 6.508
Ep 3 (Step 000025): Train loss 5.369, Val loss 6.378
Every effort moves you, and to the of the of the picture. Gis.                                     
Ep 4 (Step 000030): Train loss 4.830, Val loss 6.263
Ep 4 (Step 000035): Train loss 4.586, Val loss 6.285
Every effort moves you of the "I the picture.                    "I"I the picture"I had the picture"I the picture and I had been the picture of
Ep 5 (Step 000040): Train loss 3.879, Val loss 6.130
Every effort moves you know he had been his pictures, and I felt it's by his last word.                   "Oh, and he had been the end, and he had been
```

```
from supplementary import plot_losses

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```

<img width="504" alt="image" src="https://github.com/user-attachments/assets/71b7603a-2429-4f35-91fc-04b746e075fa" />

* In the results, we can search for some text they generate and we can see that the model memorizes the training context.

* Also note that the overfitting here occurs because we have a very, very small training set, and we iterate over it so many times
* The LLM training here primarily serves educational purposes; we mainly want to see that the model can learn to produce coherent text
* Instead of spending weeks or months on training this model on vast amounts of expensive hardware, we load pretrained weights later

## Exercise 1: Generate text from the pretrained LLM

```
start_context = 'There is always light'
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(start_context, tokenizer).to(device), 
    max_new_tokens=20, 
    context_size=1024)

print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
# There is always light I felt to the end, and in the
```

```
start_context = 'After a year of working hard'
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(start_context, tokenizer).to(device), 
    max_new_tokens=20, 
    context_size=1024)

print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
# After a year of working hard to me, and he had been--I had been his last I felt to me--his of
```

## Exercise 2: Load the pretrained model in a new session

* Open a new Python session or Jupyter notebook and load the model there

* Load the pre-trained weight we have trained before.
```
import torch

weights = torch.load('model.pth')
```

```
from supplementary import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model = model.to(device)
```

```
from supplementary import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)
import tiktoken

start_context = 'After a year of working hard'
tokenizer = tiktoken.get_encoding('gpt2')
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(start_context, tokenizer).to(device), 
    max_new_tokens=20, 
    context_size=1024)

print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
'''
 After a year of working hardleground embarrassingMen bandwidth Ortiz Dying Shareologists UFOs whorganisms ratsattron PSPloss congressionalzykEng iteration
'''
# Note that this model doesn't use weight yet!
```

* Now we use the weight that we have trained:

```
model.load_state_dict(weights)

token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids(start_context, tokenizer).to(device), 
    max_new_tokens=20, 
    context_size=1024)

print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
'''
After a year of working hard to me--I looked of the
'''
```

# 5. Loading pretrained weights (part 1)

* We will be using GPT-2 weights provided by OpenAI.

```
from importlib.metadata import version

pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")
```

* We're going to use tensorflow to load weight as OpenAI use tensorflow.

```
print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))
'''
TensorFlow version: 2.18.0
tqdm version: 4.67.1
'''
```

```
# Relative import from the gpt_download.py contained in this folder
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```
* settings is just configurations similar to what we have done before.
* params contains weights for transformer block, positional embedding layers, token embedding layers etc.

<img width="890" alt="image" src="https://github.com/user-attachments/assets/07e22c7e-7cb9-44f8-9435-11ba6bb6ec2f" />

```
print("Settings:", settings)
# Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}

print("Parameter dictionary keys:", params.keys())
# Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
'''
[[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
   0.04531523]
 [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
   0.04318958]
 [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
  -0.08785918]
 ...
 [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
  -0.06952604]
 [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
  -0.02245961]
 [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
   0.12067825]]
Token embedding weight tensor dimensions: (50257, 768)
'''
```

```
len(params['blocks']) # 12 transformer block
params['blocks'][0].keys() # keys because it is dictionary
# dict_keys(['attn', 'ln_1', 'ln_2', 'mlp'])
params['blocks'][0]['mlp'].keys()
# dict_keys(['c_fc', 'c_proj'])
params['blocks'][0]['mlp']['c_fc']
'''
{'b': array([ 0.03961948, -0.08812532, -0.14024909, ..., -0.24903017,
        -0.07680877,  0.01426962], dtype=float32),
 'w': array([[ 0.09420195,  0.09824043, -0.03210221, ..., -0.17830218,
          0.14739013,  0.07064556],
        [-0.12645245, -0.06710183,  0.03051439, ...,  0.19662377,
         -0.1203471 , -0.06284904],
        [ 0.04961119, -0.03729444, -0.04833835, ...,  0.06554844,
         -0.07141103,  0.08258601],
        ...,
        [ 0.0480442 ,  0.1574535 ,  0.00142291, ..., -0.3986896 ,
          0.08889695,  0.02402452],
        [ 0.03240572,  0.12492938, -0.04256118, ..., -0.19337358,
          0.12716232, -0.04046172],
        [-0.0316235 ,  0.00099418, -0.04906889, ..., -0.0406176 ,
          0.05363071,  0.18956499]], dtype=float32)}
'''
params['blocks'][0]['mlp']['c_fc']['w'].shape # (768, 3072)
```
* Alternatively, "355M", "774M", and "1558M" are also supported model_size arguments.

<img width="693" alt="image" src="https://github.com/user-attachments/assets/20d73113-eabd-4a29-830b-af0db3fec955" />

* We loaded the 124M GPT-2 model weights into Python, however we still need to transfer them into our GPTModel instance.

* Note that we set our context length to be smaller than that of the GPT model from 256.
* We have to change that to match the values used in training GPT 2.

```
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}


# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
```

```
from supplementary import GPTModel

gpt = GPTModel(NEW_CONFIG)
gpt.eval();
```

* The next task is to assign the OpenAI weights to the corresponding weight tensors in our GPTModel instance

```
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```

* We have to transfer weight into our model.

```
import torch
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    
load_weights_into_gpt(gpt, params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device);
```

* We can now use the model
```
import tiktoken
from supplementary import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)


tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

```
Output text:
 Every effort moves you forward.

The first step is to understand
```

## Exercise 1: Trying larger LLMs
* Load one of the larger LLMs and see how the output quality compares
* Ask it to answer specific instructions, for example to summarize text or correct the spelling of a sentence

```
# Copy the base configuration and update with specific model settings
model_name = "gpt2-medium (355M)"  # Example model name
```
* We have to download the model as well.
* We don't need to do anything extra.

# 5) Loading pretrained weights (part 2; using LitGPT)

```
from importlib.metadata import version

pkgs = ["litgpt", 
        "torch",
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")
```

* Get the lsit of LLM models:

```
!litgpt download list
```

```
Please specify --repo_id <repo_id>. Available values:
allenai/OLMo-1B-hf
allenai/OLMo-7B-hf
allenai/OLMo-7B-Instruct-hf
BSC-LT/salamandra-2b
BSC-LT/salamandra-2b-instruct
BSC-LT/salamandra-7b
BSC-LT/salamandra-7b-instruct
codellama/CodeLlama-13b-hf
codellama/CodeLlama-13b-Instruct-hf
codellama/CodeLlama-13b-Python-hf
codellama/CodeLlama-34b-hf
codellama/CodeLlama-34b-Instruct-hf
codellama/CodeLlama-34b-Python-hf
codellama/CodeLlama-70b-hf
codellama/CodeLlama-70b-Instruct-hf
codellama/CodeLlama-70b-Python-hf
codellama/CodeLlama-7b-hf
codellama/CodeLlama-7b-Instruct-hf
codellama/CodeLlama-7b-Python-hf
EleutherAI/pythia-1.4b
EleutherAI/pythia-1.4b-deduped
EleutherAI/pythia-12b
EleutherAI/pythia-12b-deduped
EleutherAI/pythia-14m
EleutherAI/pythia-160m
EleutherAI/pythia-160m-deduped
EleutherAI/pythia-1b
EleutherAI/pythia-1b-deduped
EleutherAI/pythia-2.8b
EleutherAI/pythia-2.8b-deduped
EleutherAI/pythia-31m
EleutherAI/pythia-410m
EleutherAI/pythia-410m-deduped
EleutherAI/pythia-6.9b
EleutherAI/pythia-6.9b-deduped
EleutherAI/pythia-70m
EleutherAI/pythia-70m-deduped
garage-bAInd/Camel-Platypus2-13B
garage-bAInd/Camel-Platypus2-70B
garage-bAInd/Platypus-30B
garage-bAInd/Platypus2-13B
garage-bAInd/Platypus2-70B
garage-bAInd/Platypus2-70B-instruct
garage-bAInd/Platypus2-7B
garage-bAInd/Stable-Platypus2-13B
google/codegemma-7b-it
google/gemma-2-27b
google/gemma-2-27b-it
google/gemma-2-2b
google/gemma-2-2b-it
google/gemma-2-9b
google/gemma-2-9b-it
google/gemma-2b
google/gemma-2b-it
google/gemma-7b
google/gemma-7b-it
HuggingFaceTB/SmolLM2-1.7B
HuggingFaceTB/SmolLM2-1.7B-Instruct
HuggingFaceTB/SmolLM2-135M
HuggingFaceTB/SmolLM2-135M-Instruct
HuggingFaceTB/SmolLM2-360M
HuggingFaceTB/SmolLM2-360M-Instruct
...
```
* We can then download an LLM via the following command.

```
!litgpt download microsoft/phi-2
```
<img width="778" alt="image" src="https://github.com/user-attachments/assets/cf45f382-ce0c-4e7e-9e0c-f60b718f6127" />

* A python API to use the model
```
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")

llm.generate("What do Llamas eat?")
```
* Output:
```
' Llamas are herbivores and mostly feed on grasses, hay, and other vegetation. They have a unique digestive system that allows them to extract nutrients from the tough, fibrous plants. Llamas are social animals, living in groups'
```

* We can also do below: ?
```
result = llm.generate("What do Llamas eat?", stream=True, max_new_tokens=200)
for e in result:
    print(e, end="", flush=True)
```
> * What is the diet of Llamas?
> * While there is some disagreement about what Llamas eat, they are known to consume a variety of plants and grass. They are also known to eat some small animals such as insects and insects.

* We can delete llm instance and clear memory usage.
```
del llm # to delete llm instance and clear memory
```

# 6. Instruction finetuning

<img width="763" alt="image" src="https://github.com/user-attachments/assets/db17a2ae-2289-4b9b-9e29-798a6b2bfa49" />

There are multiple ways that we can finetune the LLM model:
* Classification finetuning
* Instruction finetuning: where we provide dataset with set of instructions that the LLM follows the instruction for example fix grammar in sentence or convert from active to passive voice etc.

* We're going to use litgpt.
* A pretrained LLM is good at text completion, but it is not good at following instructions
* We need to teach the LLM to follow instructions better
* We're going to do instruction fine tuning.

<img width="550" alt="image" src="https://github.com/user-attachments/assets/0a184629-122d-4026-ac17-cb4d428f963d" />

* The left hand side are the instructions that we're gonna use to train the model.
* The LLM should response in a similar way on the right hand side.

## 6.2 Preparing a dataset for supervised instruction finetuning

* This is the training data: they include instruction key, input and we hope the LLMs will generate something similar to the output.
* We see that some don't have inputs.
* We can also get rid of the input and add it to instruction.
* This form of instruction, input and output are the historic reasons where people usually do like this.
* This example has about 1100 instructions.
* The 1000 instructions with high quality still perform well. Alpaca has 52,000 instructions.

```
[
    {
        "instruction": "Evaluate the following phrase by transforming it into the spelling given.",
        "input": "freind --> friend",
        "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."
    },
    {
        "instruction": "Edit the following sentence for grammar.",
        "input": "He go to the park every day.",
        "output": "He goes to the park every day."
    },
    {
        "instruction": "Convert 45 kilometers to meters.",
        "input": "",
        "output": "45 kilometers is 45000 meters."
    },
    {
        "instruction": "Rewrite this sentence to start with 'Although': Despite the rain, they went for a walk.",
        "input": "",
        "output": "Although it was raining, they went for a walk."
    },
    {
        "instruction": "What are the first 10 square numbers?",
        "input": "",
        "output": "1, 4, 9, 16, 25, 36, 49, 64, 81, 100."
    },
    {
        "instruction": "Suggest a more formal synonym for \"happy.\"",
        "input": "",
        "output": "A more formal synonym for \"happy\" is \"content.\""
    },
    {
        "instruction": "Translate the following sentence into French.",
        "input": "Where is the nearest restaurant?",
        "output": "O\u00f9 est le restaurant le plus proche?"
    },
   ...
```

* Load the instruction file
```
import json

file_path = "instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))
# Number of entries: 1100
```

* Each item in the data list we loaded from the JSON file above is a dictionary in the following form.

```
print("Example entry:\n", data[0])

Example entry:
 {'instruction': 'Evaluate the following phrase by transforming it into the spelling given.', 'input': 'freind --> friend', 'output': 'The spelling of the given phrase "freind" is incorrect, the correct spelling is "friend".'}
```

* **Instruction finetuning is often referred to as "supervised instruction finetuning"** because it involves training a model on a dataset where the input-output pairs are explicitly provided
* There are different ways to format the entries as inputs to the LLM; the figure below illustrates two example formats that were used for training the Alpaca (https://crfm.stanford.edu/2023/03/13/alpaca.html) and Phi-3 (https://arxiv.org/abs/2404.14219) LLMs, respectively

<img width="844" alt="image" src="https://github.com/user-attachments/assets/3a4dffaf-1859-4172-9e31-72004258aec3" />

* The phi style is shorter and less resource compare to alpaca-style.
* Alpaca style uses more resources and more memory.
* Suppose we use Alpaca-style prompt formatting, which was the original prompt template for instruction finetuning
* Shown below is how we format the input that we would pass as input to the LLM.
* As we know, when we pass input to LLM, it's more of natural language.

```
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text
```

* A formatted response with input fields.

```
model_input = format_input(data[0])
desired_response = f"\n\n### Response:\n{data[0]['output']}"

print(model_input + desired_response)
```

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Evaluate the following phrase by transforming it into the spelling given.

### Input:
freind --> friend

### Response:
The spelling of the given phrase "freind" is incorrect, the correct spelling is "friend".
```

* Another example.

```
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)
```

* In this case, input is "". That's why it won't print out.
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is an antonym of 'complicated'?

### Response:
An antonym of 'complicated' is 'simple'.
```

* Then, we tokenize the input, see chapter 2.

<img width="1021" alt="image" src="https://github.com/user-attachments/assets/720a6a55-7a7c-4253-8cef-0e431ae6927d" />

* In instruction finetuning, instruction input will have different length. If we recall, when we train/fine tune the model, we will have the same length of input.
* **We solve this issue using padding.**
* Note that we also use <|endoftext|> token id = 50256.

<img width="812" alt="image" src="https://github.com/user-attachments/assets/c5444477-b50d-4dbe-b490-dfdd7a424cd5" />

* We have 3 instruction inputs in the example. We make them all equal length by pad them to have the same length.
* Each batch doesn't have the same size, but inputs in the same batch must have the same size.
* Input instruction 1,2,3 have the same length e.g. 5 tokens. Input instruction 4,5,6 have the same length e.g. 4 tokens.
* For the target, it's the same that we shift by one.
* This is when we train as normal text and below is when we train instruction input.

<img width="459" alt="image" src="https://github.com/user-attachments/assets/250d765a-fc06-4a80-a71f-a368b355daf8" />

<img width="611" alt="image" src="https://github.com/user-attachments/assets/4e63f2c1-e4f9-4deb-9b66-f8e785ef1e75" />

* If target is shorter than input, we will have to pad the target to have the same length as input.

* In addition, it is also common to mask the target text
* By default, PyTorch has the **cross_entropy(..., ignore_index=-100) setting to ignore examples corresponding to the label -100**
* Using this -100 ignore_index, we can ignore the additional end-of-text (padding) tokens in the batches that we used to pad the training examples to equal length
* However, **we don't want to ignore the first instance of the end-of-text (padding) token (50256) because it can help signal to the LLM when the response is complete**.

* We often mask out the input to exclude them so that we only compute loss on the response.
* We mark out by adding it as -100.
<img width="737" alt="image" src="https://github.com/user-attachments/assets/55db2267-a064-4a67-8e89-ffc64b2ccd23" />


# 6. Instruction finetuning (part 2; finetuning)

* We will be working on the actual finetuning part
* We'll discuss a technique, called LoRA, that makes the finetuning more efficient
* It's not required to use LoRA, but it can result in noticeable memory savings while still resulting in good modeling performance.

## 6.1 Introduction to LoRA (Low-rank adaptation)
* Low-rank adaptation (LoRA) is a machine learning technique that **modifies a pretrained model to better suit a specific, often smaller, dataset by adjusting only a small, low-rank subset of the model's parameters**
* This approach is important because it allows for efficient finetuning of large models on task-specific data, significantly reducing the computational cost and time required for finetuning

* Typically, in regular finetuning, we have pre-trained weight W and we have updated weight delta_W which will update to the weight in each training pass.
* For example, if we have 10 weights update delta_W will be 10 as well.
* If the update is 10x10. In LoRA, the update will be, for example, 10x2 and 2x10 --> 10x10. This number 2 is a hyperparameters which we have to select.
* We save the weight by using 2 smaller matrices. 

<img width="725" alt="image" src="https://github.com/user-attachments/assets/940f5b78-bcb3-4d32-954c-770f9e9aa382" />

<img width="649" alt="image" src="https://github.com/user-attachments/assets/a4e85a80-c197-494f-b373-5ab4914b5fa8" />

* LoRA usually apply to linear layer.

<img width="671" alt="image" src="https://github.com/user-attachments/assets/1a58da35-7089-49ad-9bbd-1d45e7654448" />

## 6.2 Creating training and test sets

* There's one more thing before we can start finetuning: creating the training and test subsets
* We will use 85% of the data for training and the remaining 15% for testing
* We also have validation dataset from litgpt

```
import json

file_path = "instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))
# Number of entries: 1100
```

```
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.15)    # 15% for testing

train_data = data[:train_portion]
test_data = data[train_portion:]

print("Training set length:", len(train_data)) # 935
print("Test set length:", len(test_data)) # 165
```
* We also save into files.

```
with open("train.json", "w") as json_file:
    json.dump(train_data, json_file, indent=4)
    
with open("test.json", "w") as json_file:
    json.dump(test_data, json_file, indent=4)
```

## 6.3 Instruction finetuning

* Using LitGPT, we can finetune the model via `litgpt finetune model_dir`
* However, here, we will use **LoRA finetuning** `litgpt finetune_lora model_dir` since it will be quicker and less resource intensive

```
!litgpt finetune_lora microsoft/phi-2 \
--data JSON \
--data.val_split_fraction 0.1 \
--data.json_path train.json \
--train.epochs 3 \
--train.log_interval 100 
```
* Note that: microsoft/phi-2: Phi-2 is a Transformer with 2.7 billion parameters.
* In our machine, we don't have CUDA.
* From litgpt site, to fine-tune on MPS (the GPU on modern Macs), you can run `--precision 32-true`
```
litgpt finetune_full tiiuae/falcon-7b \
  --data Alpaca \
  --out_dir out/full/my-model-finetuned \
  --precision 32-true
```
* Note that mps as the accelerator will be picked up automatically by Fabric when running on a modern Mac.

* **Using mps on mac**
```
!litgpt finetune_lora microsoft/phi-2 \
--data JSON \
--data.val_split_fraction 0.1 \
--data.json_path train.json \
--train.epochs 3 \
--train.log_interval 100 \
--precision 32-true
```

<img width="934" alt="image" src="https://github.com/user-attachments/assets/a48f65b9-b602-4790-a951-631d044086b2" />

```
{'access_token': None,
 'checkpoint_dir': PosixPath('checkpoints/microsoft/phi-2'),
 'data': JSON(json_path=PosixPath('train.json'),
              mask_prompt=False,
              val_split_fraction=0.1,
              prompt_style=<litgpt.prompts.Alpaca object at 0x29c7a2e60>,
              ignore_index=-100,
              seed=42,
              num_workers=4),
 'devices': 1,
 'eval': EvalArgs(interval=100,
                  max_new_tokens=100,
                  max_iters=100,
                  initial_validation=False,
                  final_validation=True,
                  evaluate_example='first'),
 'logger_name': 'csv',
 'lora_alpha': 16,
 'lora_dropout': 0.05,
 'lora_head': False,
 'lora_key': False,
 'lora_mlp': False,
 'lora_projection': False,
 'lora_query': True,
 'lora_r': 8,
 'lora_value': True,
 'num_nodes': 1,
 'optimizer': 'AdamW',
 'out_dir': PosixPath('out/finetune/lora'),
 'precision': '32-true',
 'quantize': None,
 'seed': 1337,
 'train': TrainArgs(save_interval=1000,
                    log_interval=20,
                    global_batch_size=16,
                    micro_batch_size=1,
                    lr_warmup_steps=100,
                    lr_warmup_fraction=None,
                    epochs=1,
                    max_tokens=None,
                    max_steps=None,
                    max_seq_length=None,
                    tie_embeddings=None,
                    max_norm=None,
                    min_lr=6e-05)}
Seed set to 1337
Number of trainable parameters: 2,621,440
Number of non-trainable parameters: 2,779,683,840
```
* **It looks like we are using CPU?**

## Exercise 1: Generate and save the test set model responses of the base model

* We are collecting the model responses on the test dataset so that we can evaluate them later
* Starting with the original model before finetuning, load the model using the LitGPT Python API (LLM.load ...)
* Then use the LLM.generate function to generate the responses for the test data
* The following utility function will help you to format the test set entries as input text for the LLM

```
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

print(format_input(test_data[0]))
```

```
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")
```

```
from tqdm import tqdm

for i in tqdm(range(len(test_data))):
    response = llm.generate(test_data[i])
    test_data[i]["base_model"] = response
```

* Using this utility function, generate and save all the test set responses generated by the model and add them to the `test_set`
* For example, if test_data[0] entry is as follows before:
```
{'instruction': 'Rewrite the sentence using a simile.',
 'input': 'The car is very fast.',
 'output': 'The car is as fast as lightning.'}
```

* Modify the test_data entry so that it contains the model response:
```
{'instruction': 'Rewrite the sentence using a simile.',
 'input': 'The car is very fast.',
 'output': 'The car is as fast as lightning.',
  # Below is the result from the model
 'base_model': 'The car is as fast as a cheetah sprinting across the savannah.'
}
```
Do this for all test set entries, and then save the modified test_data dictionary as test_base_model.json

## Exercise 2: Generate and save the test set model responses of the finetuned model

```
from litgpt import LLM

del llm # make sure we delete the old one first
llm2 = LLM.load("out/finetune/lora/final/")
```

```
from tqdm import tqdm

for i in tqdm(range(len(test_data))):
    response = llm2.generate(test_data[i])
    test_data[i]["finetuned_model"] = response
```
<img width="925" alt="image" src="https://github.com/user-attachments/assets/3a110596-506b-4255-be80-849d77cbace1" />

# 6) Instruction finetuning (part 3; benchmark evaluation)

* In the previous notebook, we finetuned the LLM; in this notebook, **we evaluate it using popular benchmark methods**.
* There are 3 main types of model evaluation
    * MMLU-style Q&A
    * LLM-based automatic scoring
    * Human ratings by relative preference

<img width="691" alt="image" src="https://github.com/user-attachments/assets/4e7d723a-d191-4bba-a74c-815bfcdee1f1" />
<img width="688" alt="image" src="https://github.com/user-attachments/assets/2891e135-7801-4b63-ac2f-655e56c8459a" />
<img width="706" alt="image" src="https://github.com/user-attachments/assets/b454bd26-b3fb-4178-81bb-db99750c20f0" />

* Multi-choice might not be the best in evaluating the model as it measures the knowledge not the ability to generate text.

<img width="727" alt="image" src="https://github.com/user-attachments/assets/1a8aaa60-22e2-44c3-b6e3-ca3c208967ee" />
* This is to compare the response with gpt model. It's a relative benchmark.
* https://tatsu-lab.github.io/alpaca_eval/

<img width="720" alt="image" src="https://github.com/user-attachments/assets/e43c56cd-b016-4df6-97a2-1e5c486ecd12" />
* This is done by using human evalution
* https://chat.lmsys.org

<img width="842" alt="image" src="https://github.com/user-attachments/assets/750f9907-9420-4d84-abf8-918e0e1ea91e" />

## Exercise 3: Evaluate the finetuned LLM

* How to evaluate the model using MMLU

```
# List all the evaluation task
!litgpt evaluate list | grep mmlu
!litgpt evaluate out/finetune/lora/final --tasks "mmlu_philosophy" --batch_size 4
```

# 6) Instruction finetuning (part 4; evaluating instruction responses locally using a Llama 3 model)

* This notebook uses an 8 billion parameter Llama 3 model through LitGPT to evaluate responses of instruction finetuned LLMs based on a dataset in JSON format that includes the generated model responses, for example:

```
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "response_before": "\nThe atomic number of helium is 3.0", # <-- Response by an LLM
    "response_after": "\nThe atomic number of helium is 2."    # <-- Response by a 2nd LLM (finetuned model)
},
```

* I don't have this dataset.
* The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)

```
from importlib.metadata import version

pkgs = ["tqdm",    # Progress bar
        ]

for p in pkgs:
    print(f"{p} version: {version(p)}")
```

* I don't have this dataset
```
import json

json_file = "test_response_before_after.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))
```
* The structure of this file is as follows, where we have the given response in the test dataset ('output') and responses by two different models ('response_before' and 'response_after'):

```
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text

print(format_input(json_data[0])) # input
```

* Now, let's try LitGPT to compare the model responses (we only evaluate the first 5 responses for a visual comparison)

```
from litgpt import LLM
llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
```

```
from tqdm import tqdm


def generate_model_scores(json_data, json_key):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = llm.generate(prompt, max_new_tokens=50)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores
```

* Compute the score

```
scores = generate_model_scores(json_data, json_key='response_before')
```

* Compute the accuracy

```
sum(scores) / len(scores)
```
