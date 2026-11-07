# ChatThai — คำสั่งตรวจสอบ



## 1. ตรวจ Library
```...
python -m pip check
```
ต้องขึ้น `No broken requirements found`

## 2. ตรวจ GPU
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## 3. ตรวจ Hugging Face
```powershell
hf auth whoami
```


## 4. ตรวจไฟล์ Dataset
```powershell
Get-ChildItem data\raw, data\processed | Select-Object Name, Length
```
ต้องมี 4 ไฟล์ (`chat_5.txt`, `sentence2.txt`, `words.txt`, `chat_dataset.jsonl`)



## 7. Pretraining
```powershell
python -m training.pretrain --help
```
เทรน 10,000 รอบ ประเมินผลทุก 250 รอบ ดูว่า loss ลดลงเรื่อยๆ


## 9. SFT
```powershell
python -m training.sft_train --help
```
ดู arguments ก่อน แล้วเทรน จะได้ `best_pretrain.pt`

## 10. เปิด Chatbot
```powershell
python -m training.generate
```
