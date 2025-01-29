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
'''
Targets batch 1:  effort moves you
Output batch 1:  Armed heNetflix
'''

# Print the initial softmax probability scores
# probas.shape: torch.Size([2, 3, 50257])
#                          [text_idx, token_idx, vocab_id]
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probas_1)

'''
Text 1: tensor([7.4534e-05, 3.1060e-05, 1.1564e-05])
Text 2: tensor([7.4534e-05, 3.1060e-05, 1.1564e-05])
'''

# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
'''
tensor([ -9.5043, -10.3796, -11.3676, -11.4798,  -9.7765, -12.2561])
'''

# Compute the average of the score
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas) 
# tensor(-10.7940)

# Compute the negative average of the score
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# tensor(10.7940)

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

# Computing per-plexity
perplexity = torch.exp(loss)
print(perplexity)
