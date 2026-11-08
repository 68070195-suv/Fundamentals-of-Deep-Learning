# ChatAI
ChatAI คือโปรเจกต์พัฒนาแชตบอตภาษาไทยด้วย PyTorch โดยใช้สถาปัตยกรรม Mini-GPT (Transformer-based Language Model) มีเป้าหมายเพื่อให้โมเดลเรียนรู้รูปแบบภาษาไทยและสามารถสร้างคำตอบในรูปแบบบทสนทนาได้

กระบวนการพัฒนาประกอบด้วยการเตรียม Dataset, Tokenizer, Pretraining, Supervised Fine-Tuning (SFT) และการนำโมเดลไปใช้งานผ่าน Chatbot

## Task & Project Overview

### Model Architecture

| รายการ | ค่า |
|--------|-----|
| Embedding Size | 384 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Parameters | ~3,000,000 (โดยประมาณ) |
| Framework | PyTorch |
| GPU | NVIDIA RTX 5060 |

### Dataset Sources

| ไฟล์ | รายละเอียด |
|------|------------|
| `chat_5.txt` | เป็น Dataset ที่สังเคราะห์ขึ้นมา |
| `sentence2.txt` | เป็น Dataset ประโยคภาษาไทย จาก `[wannaphong/cc100-th](https://huggingface.co/datasets/wannaphong/cc100-th/blob/main/cc100-th.txt.xz)` |
| `words.txt` | เป็น Dataset คำศัพท์ภาษาไทย จาก `https://huggingface.co/datasets/ZombitX64/Thai-corpus-word/blob/main/combined_thai_corpus.txt` |
| `chat_dataset.jsonl` | เป็น Dataset เก็บบทสนทนาสำหรับฝึกได้มาจาก `chat_5.txt` |

### Project Structure

```
chatAI/
├── data/
│   ├── raw/
│   │   ├── chat_5.txt
│   │   ├── sentence2.txt
│   │   └── words.txt
│   └── processed/
│       └── chat_dataset.jsonl
├── download/
│   └── download_data.py
├── tokenizer/
│   └── tokenizer.json
│   └── train_tokenizer.py
├── model/
│   └── attention.pyy
│   └── embedding.py
│   └── feed_forward.py
│   └── gpt.py
│   └── transformer.py
├── training/
│   └── dataset.py
│   └── pretrain.py
│   └── sft_dataset.py
│   └── sft_train.py
├── checkpoints/
├── chatbot/
│   └── generate.py
└── config.py
```

---

## ติดตั้ง Library
```
python -m pip install -r requirements.txt
```


## download Dataset
```
python -m download.download_data
```


## ตรวจไฟล์ Dataset
```
Get-ChildItem data\raw, data\processed | Select-Object Name, Length
```
ต้องมี 4 ไฟล์ (`chat_5.txt`, `sentence2.txt`, `words.txt`, `chat_dataset.jsonl`)



## Pretraining
```
python -m training.pretrain --help
```
จะได้ `best_pretrain.pt`


## SFT
```
python -m training.sft_train --help
```
ดู arguments ก่อน แล้วเทรน จะได้ `best_sft.pt`

## 10. เปิด Chatbot
```
python -m chatbot.generate
```
