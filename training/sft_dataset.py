import json
import random
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from config import (
    BLOCK_SIZE,
    TOKENIZER_PATH,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE
)


class SFTDataset(Dataset):

    def __init__(self, split="train", seed=42):

        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_PATH)
        )

        dataset_path = "data/processed/chat_dataset.jsonl"

        conversations = []

        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    conversations.append(json.loads(line))

        # Shuffle and Chunk
        rng = random.Random(seed)
        rng.shuffle(conversations)

        train_end = int(len(conversations) * 0.8)
        val_end = int(len(conversations) * 0.9)

        if split == "train":
            selected_conversations = conversations[:train_end]

        elif split == "val":
            selected_conversations = conversations[train_end:val_end]

        else:
            selected_conversations = conversations[val_end:]

        self.samples = []

        for conversation in selected_conversations:

            token_ids = [BOS_TOKEN_ID]
            labels = [-100]

            for message in conversation["messages"]:

                role = message["role"]
                content = message["content"]

                if role == "user":
                    role_id = USER_TOKEN_ID
                elif role == "assistant":
                    role_id = ASSISTANT_TOKEN_ID
                elif role == "system":
                    role_id = SYSTEM_TOKEN_ID
                else:
                    continue

                content_ids = self.tokenizer.encode(
                    content,
                    add_special_tokens=False
                ).ids

                token_ids.append(role_id)
                labels.append(-100)

                token_ids.extend(content_ids)

                if role == "assistant":
                    labels.extend(content_ids)
                else:
                    labels.extend([-100] * len(content_ids))

                if role == "assistant":
                    token_ids.append(EOS_TOKEN_ID)
                    labels.append(EOS_TOKEN_ID)

            # Chunk
            for start in range(0, len(token_ids) - 1, BLOCK_SIZE):

                x = token_ids[start:start + BLOCK_SIZE]

                y = labels[start + 1:start + BLOCK_SIZE + 1]

                if not any(label != -100 for label in y):
                    continue

                padding_length = BLOCK_SIZE - len(x)

                x.extend([PAD_TOKEN_ID] * padding_length)
                y.extend([-100] * (BLOCK_SIZE - len(y)))

                self.samples.append((
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                ))

        if not self.samples:
            raise ValueError(f"{split} dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
