from pathlib import Path
import torch


BASE_DIR = Path(__file__).parent

TOKENIZER_PATH = BASE_DIR / "tokenizer" / "trained" / "tokenizer.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


VOCAB_SIZE = 8000

BLOCK_SIZE = 256

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6

DROPOUT = 0.1

# Training configuration
MAX_ITERS = 10000

EVAL_INTERVAL = 250
EVAL_ITERS = 20
LOG_INTERVAL = 50

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

GRAD_CLIP = 1.0

# SFT configuration
SFT_MAX_ITERS = 10000

SFT_BATCH_SIZE = 4
SFT_LEARNING_RATE = 5e-5

SFT_EVAL_INTERVAL = 250
SFT_EVAL_ITERS = 20
SFT_LOG_INTERVAL = 50

# Special token IDs
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
SYSTEM_TOKEN_ID = 4
USER_TOKEN_ID = 5
ASSISTANT_TOKEN_ID = 6

print("========== GPT CONFIG ==========")
print("Device:", DEVICE)
print("Vocabulary Size:", VOCAB_SIZE)
print("Block Size:", BLOCK_SIZE)
print("Embedding:", N_EMBD)
print("Attention Heads:", N_HEAD)
print("Transformer Layers:", N_LAYER)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")
