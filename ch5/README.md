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

* The flow from input text to LLM generated text is outlined in the figure below:
> 1. Given vocabulary of only 7 tokens {text : id}, map the input text to token ids.
> 2. Passing the input token ids to the GPT model, obtain 7-dim probability row vector for each input token via softmax()
> 3. Locate the index position with highest probability in each row via argmax().
> 4. Obtain all predicted token ids as index positions with the highest probability.
> 5. Map index position back to text via the inverse vocabulary.

<img width="755" alt="image" src="https://github.com/user-attachments/assets/f5d4ae68-4f9d-4792-a8ef-2b390b99d096" />

* **Hands-on**
* **Code:** [code/run_test_section5.1.2.py](code/run_test_section5.1.2.py)

```
import tiktoken
import torch
from gpt_model import GPTModel
from generate_text_sample import *
from utility import text_to_token_ids, token_ids_to_text


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

tokenizer = tiktoken.get_encoding("gpt2")

# Two input examples with 3 token ids
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]
# The targets
targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

# Feed input to the model to obtain the probability of each token
with torch.no_grad():
    logits = model(inputs)

# Probability of each token in vocabulary
probas = torch.softmax(logits, dim=-1)
print(probas)
print(probas.shape)
'''
tensor([[[1.8852e-05, 1.5173e-05, 1.1687e-05,  ..., 2.2408e-05,
          6.9776e-06, 1.8776e-05],
         [9.1572e-06, 1.0062e-05, 7.8783e-06,  ..., 2.9089e-05,
          6.0105e-06, 1.3568e-05],
         [2.9873e-05, 8.8501e-06, 1.5740e-05,  ..., 3.5459e-05,
          1.4094e-05, 1.3524e-05]],

        [[1.2561e-05, 2.0537e-05, 1.4331e-05,  ..., 1.0387e-05,
          3.4783e-05, 1.4237e-05],
         [7.2733e-06, 1.7863e-05, 1.0565e-05,  ..., 2.1206e-05,
          1.1390e-05, 1.5557e-05],
         [2.9494e-05, 3.3606e-05, 4.1031e-05,  ..., 6.5252e-06,
          5.8200e-05, 1.3697e-05]]])
torch.Size([2, 3, 50257])
'''

# Apply argmax to get the token ids with highest probability
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

'''
Token IDs:
 tensor([[[16657],[  339],[42826]],
        [[49906],[29669],[41751]]])
'''

# Convert the token IDs back into text
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
```

* Output:
```
Targets batch 1:  effort moves you
Output batch 1:  Armed heNetflix
```

* For each of the two input texts, print the initial softmax probability socres corresponding to the target tokens.

```
# Print the initial softmax probability scores
# probas.shape: torch.Size([2, 3, 50257])
#                          [text_idx, token_idx, vocab_id]
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probas_1)
```

* The three target token ID probabilities below are low.
```
Text 1: tensor([7.4534e-05, 3.1060e-05, 1.1564e-05])
Text 2: tensor([7.4534e-05, 3.1060e-05, 1.1564e-05])
```
* We want to maximize these values, making them close to a probability of of 1.

* The goal of training an LLM is to maximize the likelihood of the correct token, which invoves increasing its probability relative to other tokens.

<img width="753" alt="image" src="https://github.com/user-attachments/assets/2e1e6179-b8da-4be2-aab6-f29e9d0c3d3e" />

* The next step is to calculate the loss. See step 4,5,6.

<img width="645" alt="image" src="https://github.com/user-attachments/assets/f37f2c1b-143c-4741-932e-e6b94a4d4b23" />

* In mathematical optimisation, it is easier to it is easier to maximize the logarithm of the probability score
than the probability score itself.
```
# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
```

* Output:
```
tensor([ -9.5043, -10.3796, -11.3676, -11.4798,  -9.7765, -12.2561])
```

* Following step 5, we combine these log prob. into a single score by computing the average.

```
# Compute the average of the score
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas) 
# tensor(-10.7940)
```

* The goal is to make this average log probability as large as possible by optimizing the model weights.
* Due to the log, the largest possible value is 0, and we are currently far away from 0.
* In deep learning, instead of maximizing the average log-probability, it's a standard convention to minimize the *negative* average log-probability value; in our case, instead of maximizing -10.7940 so that it approaches 0, in deep learning, we would minimize 10.7722 so that it approaches 0
* The value negative of -10.7940, i.e., 10.7940, is also called cross-entropy loss in deep learning

```
# Compute the negative average of the score
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# tensor(10.7940)
```

* The loss we obtained is tensor(10.7940).

* The whole process can be collapsed to use Pythoch's cross_entropy function.

```
# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape) # [2, 3, 50257

# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape) # [2, 3]


logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print("Flattened logits:", logits_flat.shape) # [6, 50257]
print("Flattened targets:", targets_flat.shape) # [6]

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss) # tensor(10.7940)
```

* Note that targets are the token IDs we want the LLM to generate.
* logits are the unscaled model outputs before they enter the softmax function to obtain the probability score.

> **Perplexity**
> Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of the words in the dataset.\
> A lower perplexity indicates that the model predictions are closer to the actual distribution.\
> `perplexity = torch.exp(loss)`, in the example, we obtain tensor(48726.6094).\
> This translates to **the model being unsure about which among 48726 tokens in the vocabulary to generate the next token.**

### 5.1.3 Calculating the training and validation set losses

* This is started by preparing the training and validation datasets that will be used to train the LLM - by minimising the cross entropy loss.
* The dataset we will used for training is [The Verdict](code/the-verdict.txt).

* **Code:** [code/run_test_section5.1.3.py](code/run_test_section5.1.3.py)
```
import tiktoken

# Load the text file
file_path = "ch5/code/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Check the number of characters and tokens in the dataset
total_characters = len(text_data)
print("Characters:", total_characters)
# Characters: 20479

tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = len(tokenizer.encode(text_data))
print("Tokens:", total_tokens)
# Tokens: 5145
```

<img width="755" alt="image" src="https://github.com/user-attachments/assets/2d8c9883-dcc0-4b1e-954b-89a2827fc07b" />

* In practice, it is beneficial to train an LLM with variable-length inputs to help the LLM to better generalize across different types of inputs when it is being used.

* To split between training and validation, we define a train_ratio of 90%, in which 90% of the data are used for training and the remaining 10% as validation data.

```
# Splitting training and testing data
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

print("Train data:", len(train_data)) # Train data: 18431
print("Validation data:", len(val_data)) # Validation data: 2048
```

> Recall the `<|endoftext|>` token is assigned to relatively large token ID.\
> The BPE tokenizer which was used to train models such as GPT-2, GPT3
> and the original model used in ChatGPT has a total vocab size of 50257,
> with <|endoftext|> being assigned the largest token ID.


```
from data import create_dataloader_v1
from config import GPT_CONFIG_124M

# Adjust context_length
GPT_CONFIG_124M["context_length"] = 256

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
    drop_last=True,
    shuffle=True,
    num_workers=0
)
```

* Here, we use batch_size=2. In practice, we typically train LLMs on a larger batch size e.g. 1024.
* As an optional check, we can iterate through the data loaders to ensure that they were created correctly.

```
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("Validation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
```

* The output below shows that we have 9 training batches with 2 samples and 256 tokens each, we have 1 validation batch with 2 samples and 256 tokens each.
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

```
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device selected: {device}")

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
```

```
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# Training loss: 10.964091618855795
# Validation loss: 10.993474960327148
```

* The loss values are relatively high because we have not trained the model yet.
  
## 5.2 Training an LLM

## 5.3  Decoding strategies to control randomness

### 5.3.1 Temperature scaling

### 5.3.2 Top-k sampling

### 5.3.3 Modifying the text generation function

## 5.4 Loading and saving model weights in PyTorch

## 5.5 Loading pretrained weights from OpenAI

## Summary
