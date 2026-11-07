# ChatThai — คำสั่งตรวจสอบ



## 1. ตรวจ Library
```
python -m pip check
```


## 2. ตรวจ GPU
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## 3. ตรวจ Hugging Face
```
hf auth whoami
```


## 4. ตรวจไฟล์ Dataset
```
Get-ChildItem data\raw, data\processed | Select-Object Name, Length
```
ต้องมี 4 ไฟล์ (`chat_5.txt`, `sentence2.txt`, `words.txt`, `chat_dataset.jsonl`)



## 7. Pretraining
```
python -m training.pretrain --help
```
จะได้ `best_pretrain.pt`


## 9. SFT
```
python -m training.sft_train --help
```
ดู arguments ก่อน แล้วเทรน จะได้ `best_sft.pt`

## 10. เปิด Chatbot
```
python -m training.generate
```
