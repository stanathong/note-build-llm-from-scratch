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

'''
Output text:
 Every effort moves you rentingetic wasnم refres RexMeCHicular stren
'''