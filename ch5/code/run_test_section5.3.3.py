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

'''
Output text:
 Every effort moves you?"" Gisburn rather a--I felt nervous and left behind enough--she's the mant was that Mrs. G
'''
