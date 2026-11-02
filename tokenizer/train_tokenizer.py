
import json
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers import decoders, normalizers


# ==========================================
# 1. กำหนด Path
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

GENERAL_TEXT_PATH = (
    ROOT_DIR / "data" / "raw" / "sentence2.txt"
)

WORD_TEXT_PATH = (
    ROOT_DIR / "data" / "raw" / "words.txt"
)

CHAT_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "chat_dataset.jsonl"
)

OUTPUT_DIR = ROOT_DIR / "tokenizer" / "trained"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ตั้งค่า Tokenizer
# ==========================================

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


# ==========================================
# 3. เตรียมข้อมูลสำหรับฝึก
# ==========================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def training_data():
    total_general = 0
    total_words = 0
    total_chat = 0

    # General Thai Text
    if not GENERAL_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ General Text: {GENERAL_TEXT_PATH}"
        )

    with open(
        GENERAL_TEXT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            text = normalize_text(line)

            if text:
                total_general += 1
                yield text

    # Word Dataset
    if not WORD_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Word Dataset: {WORD_TEXT_PATH}"
        )

    with open(
        WORD_TEXT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            text = normalize_text(line)

            if text:
                total_words += 1
                yield text

    # Conversation Dataset
    if not CHAT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Chat Dataset: {CHAT_DATA_PATH}"
        )

    with open(
        CHAT_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                conversation = json.loads(line)
                messages = conversation["messages"]

                for message in messages:
                    role = message["role"]
                    content = normalize_text(message["content"])

                    if role not in ["user", "assistant", "system"]:
                        continue

                    if content:
                        total_chat += 1
                        yield content

            except (json.JSONDecodeError, KeyError, TypeError) as error:
                print(
                    f"ข้ามข้อมูลบรรทัด {line_number}: {error}"
                )

    print("\nจำนวนข้อมูลที่อ่านได้")
    print(f"General Text: {total_general:,} บรรทัด")
    print(f"Word Dataset: {total_words:,} บรรทัด")
    print(f"Chat Messages: {total_chat:,} ข้อความ")


# ==========================================
# 4. สร้าง BPE Tokenizer
# ==========================================

def build_tokenizer():

    tokenizer = Tokenizer(
        models.BPE(unk_token="<unk>")
    )

    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    print("เริ่มฝึก BPE Tokenizer...")

    tokenizer.train_from_iterator(
        training_data(),
        trainer=trainer
    )

    # บันทึก Tokenizer
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"

    tokenizer.save(str(tokenizer_path))

    tokenizer.model.save(str(OUTPUT_DIR))

    print("\nฝึก Tokenizer เสร็จแล้ว")
    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}")
    print(f"บันทึกที่: {OUTPUT_DIR}")

    return tokenizer


# ==========================================
# 5. ทดสอบ Tokenizer
# ==========================================

def test_tokenizer(tokenizer):

    test_sentences = [
        "สวัสดีครับ วันนี้เป็นอย่างไรบ้าง",
        "ผมกำลังพัฒนา AI ภาษาไทยด้วย PyTorch",
        "ช่วยอธิบาย Machine Learning ให้หน่อย",
        "วันนี้รู้สึกเหนื่อยมาก ไม่รู้จะทำอย่างไรดี"
    ]

    print("\n========== TOKENIZER TEST ==========")

    for sentence in test_sentences:

        encoded = tokenizer.encode(sentence)

        decoded = tokenizer.decode(encoded.ids)

        print("\nข้อความ:", sentence)
        print("Tokens:", encoded.tokens)
        print("Token IDs:", encoded.ids)
        print("จำนวน Tokens:", len(encoded.ids))
        print("Decoded:", decoded)


# ==========================================
# 6. Main
# ==========================================

if __name__ == "__main__":

    tokenizer = build_tokenizer()

    test_tokenizer(tokenizer)
