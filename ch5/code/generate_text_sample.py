# ch4/code/section-4.7-generate-text-sample.py

import torch
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

        # Focus only on the last timestep
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # -1 indicates the last token
        logits = logits[:, -1, :] # [batch, vocab_size]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1) # [batch, vocab_size]

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # [batch, 1]

        #  Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=-1) # (batch, n_tokens + 1)

    return idx


