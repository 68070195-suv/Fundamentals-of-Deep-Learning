import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert N_EMBD % N_HEAD == 0

        self.head_size = N_EMBD // N_HEAD

        self.query = nn.Linear(N_EMBD, N_EMBD)
        self.key = nn.Linear(N_EMBD, N_EMBD)
        self.value = nn.Linear(N_EMBD, N_EMBD)

        self.projection = nn.Linear(N_EMBD, N_EMBD)

        self.attention_dropout = nn.Dropout(DROPOUT)
        self.residual_dropout = nn.Dropout(DROPOUT)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        # Create Query, Key, Value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split into multiple heads
        q = q.view(
            batch_size, sequence_length,
            N_HEAD, self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size, sequence_length,
            N_HEAD, self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size, sequence_length,
            N_HEAD, self.head_size
        ).transpose(1, 2)

        # Calculate attention scores
        scores = q @ k.transpose(-2, -1)

        scores = scores * (self.head_size ** -0.5)

        # Apply causal mask
        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.attention_dropout(
            attention_weights
        )

        # Weighted sum of values
        output = attention_weights @ v

        # Combine heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_size
        )

        # Final projection
        output = self.projection(output)

        output = self.residual_dropout(output)

        return output
