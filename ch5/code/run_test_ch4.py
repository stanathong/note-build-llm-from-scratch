import torch
import tiktoken

from gpt_model import GPTModel
from config import GPT_CONFIG_124M
from generate_text_sample import *


tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
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