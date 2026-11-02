import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from config import BLOCK_SIZE, TOKENIZER_PATH, BASE_DIR


RAW_TEXT_PATH = (
    BASE_DIR / "data" / "raw" /
    "sentence2.txt"
)


class PretrainingDataset(Dataset):

    def __init__(self):

        if not RAW_TEXT_PATH.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {RAW_TEXT_PATH}"
            )

        # Tokenizer
        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_PATH)
        )

        # Token ID
        token_ids = []

        with open(
            RAW_TEXT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                # text to Token ID
                ids = self.tokenizer.encode(line).ids

                # add Token 
                token_ids.extend(ids)

                # add EOS 
                token_ids.append(3)

        if len(token_ids) <= BLOCK_SIZE:
            raise ValueError(
                "Dataset is too small for BLOCK_SIZE"
            )

        # change to Tensor
        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.tokens - 1) // BLOCK_SIZE

    def __getitem__(self, index):

        start = index * BLOCK_SIZE

        # input Token
        x = self.tokens[
            start:start + BLOCK_SIZE
        ]

        # target Token
        y = self.tokens[
            start + 1:start + BLOCK_SIZE + 1
        ]

        return x, y
