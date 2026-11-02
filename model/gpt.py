import torch
import torch.nn as nn

from config import VOCAB_SIZE, N_LAYER, N_EMBD
from model.embedding import GPTEmbedding
from model.transformer import TransformerBlock


class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()

        # change Token ID to Embedding
        self.embedding = GPTEmbedding()

        # Transformer Blocks 4 Layers
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYER)]
        )

        # Normalize Last Layer
        self.final_norm = nn.LayerNorm(N_EMBD)

        # change Embedding to Token
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, token_ids):

        # Token Embedding + Positional Embedding
        x = self.embedding(token_ids)

        # ผ่าน Transformer Blocks
        x = self.blocks(x)

        # Final LayerNorm
        x = self.final_norm(x)

        # คำนวณคะแนนของ Token ทุกตัวใน Vocabulary
        logits = self.lm_head(x)

        return logits