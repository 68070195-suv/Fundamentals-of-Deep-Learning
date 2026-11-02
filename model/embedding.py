import torch
import torch.nn as nn

from config import DROPOUT, VOCAB_SIZE, BLOCK_SIZE, N_EMBD


class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=N_EMBD
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=BLOCK_SIZE,
            embedding_dim=N_EMBD
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        return self.embedding(positions)


class GPTEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = TokenEmbedding()
        self.position_embedding = PositionalEmbedding()

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, token_ids):
        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.position_embedding(token_ids)

        embeddings = token_vectors + position_vectors

        return self.dropout(embeddings)
