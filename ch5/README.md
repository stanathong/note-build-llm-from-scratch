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

* This is a typical training loop.
<img width="514" alt="image" src="https://github.com/user-attachments/assets/f4bc06ef-472e-41f3-b289-81c8560e565a" />

* **Hands-on**
* Code: [code/train_model.py](code/train_model.py)

```
import torch
from loss import calc_loss_batch, calc_loss_loader
from utility import text_to_token_ids, token_ids_to_text
from generate_text_sample import generate_text_simple

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # Resets loss gradeints from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
                
        # Print a simple text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen

# Calculate loss over the training and validation set
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # Set to evaluation model with gradient tracking and dropout disable
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Take a text snippet as input, converts to token IDs, feeds to LLM
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", "")) # Compact print format
    model.train()
```

> **AdamW**\
> When training LLM, we chose AdamW optimiser over Adam, it improves the weight decay approach,
> which aims to minimise model complexity and prevent overfitting by penalising larger weights.\
> This adjustment enables AdamW to achieve more effective regularisation and better generalisation.


* Train a GPT model instance for 10 epochs.
* Code: [code/run_test_section5.2.py](code/run_test_section5.2.py)

```
import torch
import tiktoken
from data import create_dataloader_v1
from config import GPT_CONFIG_124M
from loss import calc_loss_loader
from train_model import train_model_simple
from gpt_model import GPTModel

# Adjust context_length
GPT_CONFIG_124M["context_length"] = 256

...

# 5.2 Train the LLM

import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

* The output below shows that the training loss improve drastically.

```
Device selected: mps
Ep 1 (Step 000005): Train loss 8.833, Val loss 8.888
Ep 1 (Step 000010): Train loss 7.019, Val loss 7.465
Every effort moves you,,,,,,,,,,,,.
Ep 2 (Step 000015): Train loss 6.101, Val loss 6.693
Every effort moves you, the, the of the, the, the, the.", the, the,,, the, and, the, of the, the, the, the, the, the, the, and, the,, the,
Ep 3 (Step 000020): Train loss 5.746, Val loss 6.498
Ep 3 (Step 000025): Train loss 5.814, Val loss 6.526
Every effort moves you, and, and and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
Ep 4 (Step 000030): Train loss 5.578, Val loss 6.918
Ep 4 (Step 000035): Train loss 5.215, Val loss 6.454
Every effort moves you, and I had to the of the of the of the of the of the of theisburn, and to to the of the of the of the of the of the of the of the of the of the of the of theis, and
Ep 5 (Step 000040): Train loss 4.961, Val loss 6.437
Ep 5 (Step 000045): Train loss 4.351, Val loss 6.316
Every effort moves you know "
Ep 6 (Step 000050): Train loss 3.742, Val loss 6.195
Ep 6 (Step 000055): Train loss 3.517, Val loss 6.149
Every effort moves you know it was his a little the picture.
Ep 7 (Step 000060): Train loss 3.102, Val loss 6.099
Every effort moves you know it was not that the picture.
Ep 8 (Step 000065): Train loss 2.449, Val loss 6.127
Ep 8 (Step 000070): Train loss 2.096, Val loss 6.178
Every effort moves you know," was to have my dear, and he had been the picture."I turned, and I had been at my elbow and as his pictures, and down the room, and in
Ep 9 (Step 000075): Train loss 1.772, Val loss 6.205
Ep 9 (Step 000080): Train loss 1.423, Val loss 6.251
Every effort moves you know," was one of the picture for nothing--I told Mrs."I looked--I looked up, I felt to see a smile behind his close grayish beard--as if he had the donkey. "There were days when I
Ep 10 (Step 000085): Train loss 1.101, Val loss 6.240
Ep 10 (Step 000090): Train loss 0.749, Val loss 6.294
Every effort moves you?""Yes--quite insensible to the irony. She wanted him vindicated--and by me!""I didn't dabble back his head to look up at the sketch of the donkey. "There were days when I

Training completed in 1.31 minutes.
```

* Plot the resulting training and validation loss
* Code: [code/plot_losses.py](code/plot_losses.py)

```
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()
```

* By plotting loss:

```
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```

<img width="490" alt="image" src="https://github.com/user-attachments/assets/d00d374e-684b-4103-b1ba-b49092ec80e5" />

* The plot has shown that the losses start to diverge past the second epoch.
* This divergence and the fact that the validation loss is much larger than the training loss indicating that the model is overfitting to the training data.
* The model memorizes the training data since we're working with a very small training data, and training the model for multple epochs.
* **Usually, it's common to train a model on a much larger dataset for only one epoch.**

## 5.3  Decoding strategies to control randomness

* After training, we put the model into evaluation mode to turn off random components such as dropout. We will transfer the model back to the CPU for inference as the model is quite small.
* Code: [code/run_test_section5.3.py](code/run_test_section5.3.py)

```
model.to("cpu")
model.eval()

for i in range(3):
    token_ids = generate_text_simple(
        model, 
        idx=text_to_token_ids("Every effort moves you", tokenizer), 
        max_new_tokens=25, 
        context_size=GPT_CONFIG_124M["context_length"])

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

* Output:
```
Output text:
 Every effort moves you?"

"Yes--quite insensible to the irony. She wanted him vindicated--and by me!"


Output text:
 Every effort moves you?"

"Yes--quite insensible to the irony. She wanted him vindicated--and by me!"


Output text:
 Every effort moves you?"

"Yes--quite insensible to the irony. She wanted him vindicated--and by me!"
```

* Give the same starting context "Every effort moves you", LLM will always generate the same outputs.
* This is because the generated token is selected corresponding to the largest probability score among all tokens in the vocabulary.

### 5.3.1 Temperature scaling

* **Temporal scaling is a technique that adds a probabilistic selection process** to the next token generation task.
* Previously, we always sampled the token with the highest probability as the next token using `torch.argmax`, known as **greedy decoding**.
* To generate text with more variety, we can **replace argmax with a function that samples from a probability distribution.**
* This **probability distribution is the probability scores that LLM generate for each vocab at each token generation step**. 
* To illustrate the **probabilistic sampling** for the next token generation process:

```
vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

inverse_vocab = {v: k for k, v in vocab.item()}
```

* Assume the LLM is given the context "every effort moves you", and returns the following logits for the next token:

```
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
```

* The typical process is to **convert the logis into probabilities via the softmax** function.
* It then obtain the token ID corresponding to the generated token via `argmax`.
* We can then map the token back into text via the inverse vocabulary, inverse_vocab above.

```
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print('probas:', probas)
print('next_token_id:', next_token_id, 'text:', inverse_vocab[next_token_id])
```

* The code outputs the largest logit value, which is corresponding to the largest softmax probability score, which is the 3rd index.

```
probas: tensor([6.0907e-02, 1.6313e-03, 1.0019e-04, 5.7212e-01, 3.4190e-03, 1.3257e-04,
        1.0120e-04, 3.5758e-01, 4.0122e-03])
next_token_id: 3 text: forward
```

* **To implement a probabilistic sampling process, we can replace argmax with the multinomial function in Pytorch**

```
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print('next_token_id:', next_token_id, 'text:', inverse_vocab[next_token_id])
```

* Output:
```
next_token_id: 3 text: forward
```

* The multinomial function samples the next token proportional to its probability score.
* "forward" is still the most likely token and will be selected by multinomial most of the time but not all the time.
* This can be illustrated by repeating this sampling 1,000 times.

```
def print_sampled_tokens(probas):
    torch.manual_seed(123) # reset the seed every time
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
```
* The sampling output is

```
73 x closer
0 x every
0 x effort
582 x forward
2 x inches
0 x moves
0 x pizza
343 x toward
```

* In summary, if we replace the argmax function with the multinomail function, the LLM would not always generate the same output.

* We can **control the distribution and selection process via a concept called "temperature scaling"**.
* Temperature scaling just **divides the logits** by a number greater than 0.

```
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```
> **Note**\
> Temperatures **greater than 1 result in more uniformly** distributed token probabilities.\
> Temperatures **smaller than 1 result in more confident (sharper/more peaky)** distributed token probabilities.

```
# Plot graph with different temperatures
# 1: original, 0.1: higher confidence, 5: lower confidence
temperatures = [1, 0.1, 5]
scaled_probs = [softmax_with_temperature(next_token_logits, T)
                for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5,3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probs[i],
                   bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
```

<img width="488" alt="image" src="https://github.com/user-attachments/assets/f23c07df-3b4b-4f88-b16c-2113e50a2390" />


> Temperature = 1 divides the logits by 1 before passing them to the softmax func-
tion to compute the probability scores.\
> Using a temperature of 1 is the same as not using any temperature scaling.\
> In this case, **the tokens are selected with a probability equal to the original softmax** probability scores via the multinomial sampling function.\
> For example, for the temperature setting 1, the token corresponding to “forward” would be selected about 60% of the time.
> Applying very **small temperatures, such as 0.1, will result in sharper distributions**.\
> For such case, the multinomial function selects the most likely token (here, "forward") almost 100% of the time, approaching the behavior of the argmax function.
> A temperature of 5 results in a more uniform distribution where other tokens are selected more often.
> With a higher temperature, this can add more variety to the generated texts but also more often results in nonsensical text. 

### 5.3.2 Top-k sampling

* Temperature increases the diversity of the output, as it reduces the likelihood of the model repeatedly selecting the most probable token.
* Using temperature explores less likely but potentially more interesting and creative paths in the generation process.
* One downside of this approach:
    * it sometimes leads to grammatically incorrect or completely nonsensical outputs.
    * Ex. every effort moves you pizza.
* Top-k sampling, when combined with probabilistic sampling (torch.multinomial) and temperature scaling, can improve the text generation results.
* In top-k sampling, we restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from being selected by masking their probability scores to -inf.

<img width="752" alt="image" src="https://github.com/user-attachments/assets/19357a8d-5ddb-4b03-bd3a-a32ebb68929e" />
> Using top-k sampling with k = 3, we focus on 3 tokens with the highest logits and mask out
> all other tokens with negative infinity (–inf) before applying the softmax function.
> This results in a probability distribution with a probability value 0 assigned to all non-top-k tokens.

* **Code:** [code/run_test_section5.3.2.py](code/run_test_section5.3.2.py)

```
import torch

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

inverse_vocab = {v: k for k, v in vocab.items()}

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("Top logits:", top_logits)
print("Top positions:", top_pos)
# Get the top vocab
top_vocabs = [inverse_vocab[pos.item()] for pos in top_pos]
print("Top vocabs:", top_vocabs)
```

* The logits values and token IDs of the top three tokens, **in descending order**, are
```
Top logits: tensor([6.7500, 6.2800, 4.5100])
Top positions: tensor([3, 7, 0])
Top vocabs: ['forward', 'toward', 'closer']
```

* Then apply **PyTorch’s where function** to set the logit values of tokens that are below the lowest logit value within our top-three selection to negative infinity (-inf):

```
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)
print(new_logits)
```

* The resulting logits for the next token in the nine-token vocabulary are:
```
tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])
```

* Then apply the softmax function to turn these into next-token probabilities:
```
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
```

* The result of this top-three approach are three non-zero probability scores:
```
tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
```

* Logits -> top-k -> -inf mask -> softmax

* We can now apply the **temperature scaling and multinomial function** for probabilistic sampling to **select the next token among these three non-zero probability scores** to generate the next token. 

### 5.3.3 Modifying the text generation function

* Combine *temperature sampling and top-k sampling*:
* Replace [code/generate_text_sample.py](code/generate_text_sample.py) with [code/generate.py](code/generate.py)

```
import torch

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # last token

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float("-inf")).to(logits.device), 
                logits)
        
        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
```

* **Code:** [code/run_test_section5.3.3.py](code/run_test_section5.3.3.py)

```
import torch
import tiktoken
from data import create_dataloader_v1
from config import GPT_CONFIG_124M
from train_model import train_model_simple
from gpt_model import GPTModel
from plot_losses import plot_losses
from generate_text_sample import *
from utility import text_to_token_ids, token_ids_to_text
from generate import *


# Adjust context_length
GPT_CONFIG_124M["context_length"] = 256

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device selected: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

# Load the text file
file_path = "ch5/code/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = len(tokenizer.encode(text_data))

# Splitting training and testing data
train_ratio = 0.9
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
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# 5.2 Train the LLM

import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


# 5.3.3 Text generation
torch.manual_seed(123)

model.to("cpu")
model.eval()

token_ids = generate(
    model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer), 
    max_new_tokens=25, 
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

```
Output text:
 Every effort moves you?"" Gisburn rather a--I felt nervous and left behind enough--she's the mant was that Mrs. G
```
* The output is different from the book!

* The output from the book and the official's repo:
    * The generated text is very different from the one we previously generated via the generate_simple function ("Every effort moves you know," was one of the axioms he laid...! ), which was a memorized passage from the training set.

```
Output text:
 Every effort moves you stand to work on surprise, a one of us had gone with random-
```

### Exercise 5.1

### Exercise 5.2

### Exercise 5.3

## 5.4 Loading and saving model weights in PyTorch

### Saving a Pytorch model

* Save a model’s state_dict, a dictionary mapping each layer to its parameters.

```
torch.save(model.state_dict(), "model.pth")
```

* "model.pth" is the filename where the state_dict is saved.
* The .pth extension is a convention for PyTorch files, though we could technically use any file extension.
* After saving the model weights via the state_dict, we can load the model weights into a new GPTModel model instance:

```
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

* Optimizer, for example AdamW store additional parameters for each model weight.
* AdamW uses historical data to adjust learning rates for each model parameter dynamically.
* Without it, the optimizer resets, and the model may learn suboptimally or even fail to converge properly, which means it will lose the ability to generate coherent text.
* Using torch.save, we can save both the model and optimizer state_dict contents:

```
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)
```

* We can restore the model and optimizer states by first loading the saved data via torch.load and then using the load_state_dict method:

```
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M) model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1) optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
```

* **Code:** [code/run_test_section5.3.3.py](code/run_test_section5.3.3.py)

```
# 5.3.4 Save and Load model

print("Save the trained model")
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)

print("Load the trained model")

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

torch.manual_seed(123)

print("Regenerate output again")

model.to("cpu")
model.eval()

token_ids = generate(
    model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer), 
    max_new_tokens=25, 
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

'''
Output text:
 Every effort moves you?"" Gisburn rather a--I felt nervous and left behind enough--she's the mant was that Mrs. G
'''
```

## 5.5 Loading pretrained weights from OpenAI

## Summary
