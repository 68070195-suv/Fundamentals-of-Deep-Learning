import torch.nn as nn

from config import N_EMBD
from model.attention import CausalSelfAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        # Normalize pre Attention
        self.ln1 = nn.LayerNorm(N_EMBD)

        # Self-Attention
        self.attention = CausalSelfAttention()

        # Normalize pre Feed Forward
        self.ln2 = nn.LayerNorm(N_EMBD)

        # Feed Forward Network
        self.feed_forward = FeedForward()

    def forward(self, x):

        # Attention + Residual Connection
        x = x + self.attention(self.ln1(x))

        # Feed Forward + Residual Connection
        x = x + self.feed_forward(self.ln2(x))

        return x
