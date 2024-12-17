# Position-Agnostic in Self-Attention

The self-attention mechanism in large language models (LLMs) is inherently position-agnostic, meaning that it does not inherently account for the order of words in a sequence.

## Self-Attention + Position-Agnostic

### How Self-Attention Works:

* In self-attention, the **attention scores between tokens are computed based on their content** (e.g., similarity of their representations).
* For each token in the sequence, self-attention aggregates information from all other tokens, weighting them according to these computed attention scores.

### Key Property: No Awareness of Word Order:

* **Self-attention only considers the relationship between tokens' embeddings** but does **not know where those tokens are positioned in the sequence**.
* For example, the phrases **"The cat sat on the mat" and "The mat sat on the cat" would result in the same attention scores** if no positional information is added.
* This is because self-attention sees the embeddings but doesn't differentiate based on their sequence.

