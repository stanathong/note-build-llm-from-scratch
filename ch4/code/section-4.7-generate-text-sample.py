import torch
import tiktoken
from gpt_model import GPTModel 
from config import GPT_CONFIG_124M

def generate_text_simple(model, idx, max_new_tokens, context_size):
    '''
    idx [batch, n_tokens]: array of indices in the current context
    '''
    # Loop for # max_new_tokens required to generate
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # keep the last tokens as context
        idx_cond = idx[:, -context_size:] # [batch, context_size]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # -1 here indicates the last token 
        logits = logits[:, -1, :] # [batch, vocab_size]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1) # [batch, vocab_size]

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=-1) # (batch, n_tokens + 1)

    return idx

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
print("encoded_tensor.shape:", encoded_tensor.shape)

# The encode IDs are 
'''
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])
'''

# Put the model into .eval() mode
# This disable random components like dropout, which are only used during training.

model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# Convert the IDs back into text using the tokenizer
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

'''
Output: tensor([[15496,    11,   314,   716,  1755, 31918,  8247,  1755, 37217,  4085]])
Output length: 10
Hello, I am night Rafaelzens night travellers emot
'''
# The model generated gibberish because we haven't trained the model yet.
# The model is just initialised with initial weights.
