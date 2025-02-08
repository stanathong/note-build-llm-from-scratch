import torch
import matplotlib.pyplot as plt

# Illustration for Section 5.3.1
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

print("vocab:\n",vocab)
print("inverse vocab:\n",inverse_vocab)

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0) # torch.Size([9])
next_token_id = torch.argmax(probas).item()
print('probas:', probas)
print('next_token_id:', next_token_id, 'text:', inverse_vocab[next_token_id])

'''
probas: tensor([6.0907e-02, 1.6313e-03, 1.0019e-04, 5.7212e-01, 3.4190e-03, 1.3257e-04,
        1.0120e-04, 3.5758e-01, 4.0122e-03])
next_token_id: 3 text: forward
'''

# To implement a probabilistic sampling process, we use multinomial function
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print('next_token_id:', next_token_id, 'text:', inverse_vocab[next_token_id])

'''
next_token_id: 3 text: forward
'''

# Repeating sampling 1000 times
def print_sampled_tokens(probas):
    torch.manual_seed(123) # reset the seed every time
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

'''
73 x closer
0 x every
0 x effort
582 x forward
2 x inches
0 x moves
0 x pizza
343 x toward
'''

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

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
