# Chapter 5: Pretraining on unlabeled data

<img width="690" alt="image" src="https://github.com/user-attachments/assets/a5515921-45e8-4f88-ad49-39f11137b9a3" />

* This chapter focuses on **stage 2**, which will include the following topics:
    * Pretraining the LLM (step 4)
    * Implemetning the training code (step 5)
    * Evaluating the performance (step 6)
    * Saving and loading model weights (step 7) 

## 5.1 Evaluating generative text models

* The topics to be discussed in this chapter: Step 1, 2, and 3.

<img width="638" alt="image" src="https://github.com/user-attachments/assets/dd14820c-74af-453b-a88b-c5d522bb3a0f" />

### 5.1.1 Using GPT to generate text

* Recap the previous chapter for using GPTModel to generate text
* **Code:** [code/run_test_ch4.py](code/run_test_ch4.py)

```
import torch
import tiktoken

from gpt_model import GPTModel
from config import GPT_CONFIG_124M
from generate_text_sample import *

# Generating input for GPTModel
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am" # STARTING INPUT TEXT
encoded = tokenizer.encode(start_context)
print("encoded:", encoded) # [15496, 11, 314, 716]
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
print('encoded_tensor:', encoded_tensor.shape) # torch.Size([1, 4])

# shorten the context length from the usual 1024 to 256
GPT_CONFIG_124M["context_length"] = 256
print("GPT_CONFIG_124M:", GPT_CONFIG_124M)
'''
GPT_CONFIG_124M: {
    'vocab_size': 50257, 
    'context_length': 256, 
    'emb_dim': 768, 
    'n_heads': 12, 
    'n_layers': 12, 
    'drop_rate': 0.1, 
    'qkv_bias': False}
'''

torch.manual_seed(123)

# Put the model into .eval() mode
# This disable random components like dropout, which are only used during training.
model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
'''
Output: tensor([
[15496,    11,   314,   716, 13240, 11381,  4307,  7640, 16620, 34991,6842, 37891, 19970, 47477]])
Output length: 14
'''

# Convert the IDs back into text using the tokenizer
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Hello, I am Laur inhab DistrinetalkQueue bear confidentlyggyenium
```

* In this chapter, we'll shorten context_length from 1024 to 256 to reduces the computation demands of training the model

```
GPT_CONFIG_124M["context_length"] = 256

GPT_CONFIG_124M: {
    'vocab_size': 50257, 
    'context_length': 256, 
    'emb_dim': 768, 
    'n_heads': 12, 
    'n_layers': 12, 
    'drop_rate': 0.1, 
    'qkv_bias': False}
```

* In the code above, we have to tokenize input text to ids, then after generating text, we convert the token ids to text.
* In this function, we will instead implement them as functions: `text_to_token_ids` and `token_ids_to_text`.

* The figure below shows that generating text involves encoding text into token IDs that the LLM processes into logit vectors.
* The logit vectors are then converted back into token IDs, detokenized into a text representation.

<img width="761" alt="image" src="https://github.com/user-attachments/assets/808c9bb0-a085-4cbc-8755-160db1149b8a" />


> **Generating text using a GPT model**
> 1. Use the tokenizer to convert input text into a series of token IDs
> 2. Pass the token IDs to the model to generate logits (vectors representing the probability distribution for each token in the vocabulary)
> 3. Convert these logits back into token IDs
> 4. Use the tokenizer to decodes into human-readable text

* **Code:** [code/run_test_section5.1.py](code/run_test_section5.1.py)

```
import tiktoken
import torch
from gpt_model import GPTModel
from generate_text_sample import *

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Remove batch dimension
    return tokenizer.decode(flat.tolist()) 


# Shorten context length from 1024 to 256
GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 256,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

* Output from the model given the start context: "Every effort moves you"

```
Output text:
 Every effort moves you rentingetic wasnم refres RexMeCHicular stren
```

* The model isn't yet producing conherent text because we haven't trained it yet.
* In order to measure whether the generated text is coherent or high quality, we need *a numerical method to evaluate the generated content*.


### 5.1.2 Calculating the text generation loss

### 5.1.3 Calculating the training and validation set losses

## 5.2 Training an LLM

## 5.3  Decoding strategies to control randomness

### 5.3.1 Temperature scaling

### 5.3.2 Top-k sampling

### 5.3.3 Modifying the text generation function

## 5.4 Loading and saving model weights in PyTorch

## 5.5 Loading pretrained weights from OpenAI

## Summary
