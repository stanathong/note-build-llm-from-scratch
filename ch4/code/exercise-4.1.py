from importlib.metadata import version
from transformer_block import TransformerBlock

print("torch version:", version("torch"))

GPT_CONFIG_124M = {
    'vocab_size': 50257,        # Vocabulary size
    'context_length': 1024,     # Context length
    'emb_dim': 768,             # Embedding dimension
    'n_heads': 12,              # Number of attention heads
    'n_layers': 12,             # Number of layers
    'drop_rate': 0.1,           # Dropout rate
    'qkv_bias': False,          # Query-Key-Value bias
}

block = TransformerBlock(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in block.ff.parameters())
print(f"The total number of parameters in feed forward module is {total_params:,}")
# The total number of parameters in feed forward module is 4,722,432

total_params = sum(p.numel() for p in block.att.parameters())
print(f"The total number of parameters in attention module is {total_params:,}")
# The total number of parameters in attention module is 2,360,064



