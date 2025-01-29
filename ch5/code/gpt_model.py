import torch
import torch.nn as nn

from transformer import TransformerBlock
from components import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Embedding layer is just a look up table for given indices
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # input: in_idx is tokenised text
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape # [batch, n_tokens]
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 

