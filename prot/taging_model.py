import os
import random
import joblib
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Конфигурация
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MODEL_NAME = "cointegrated/rubert-tiny2"
BATCH_SIZE = 16
EPOCHS = 4
MAX_LEN = 320
LR = 2e-5
NUM_WORKERS = 4
USE_CLASS_WEIGHTS = False           # можно поставить True при сильном дисбалансе классов
FILE_PATH = r"tagged_texts.xlsx"    # Поменять на нужный нам файл с данными
SHEET_NAME = r"Sheet1"              # Поменять на нужным лист в таблице
TEXT_COL = r"text"
TAG_COL = r"tag"
OUTPUT_DIR = "./output"                    # Вставить, куда выгружать данные и модель

# Воспроизводимость. Делаем так, чтобы случайные значения не сбивали модель при каждой итерации, параметры и выборки будут сохранены в seed
def set_seed(seed=SEED):
    random.seed(seed)                   # Python random
    np.random.seed(seed)                # NumPy random
    torch.manual_seed(seed)             # CPU
    torch.cuda.manual_seed_all(seed)    # GPU

set_seed()

# Функция очистки данных
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)         # Удалить URL
    text = re.sub(r"@\w+", "", text)            # Удалить упоминания
    text = re.sub(r'[^\w\s]', " ", text)        # удалить спецсимволы
    text = re.sub(r"\s+", " ", text)            # Множественные пробелы -> одинарные
    return text.strip()

# Загрузка данных
df = pd.read_csv('combined_news_dataset_V2.csv', sep=',')
# Автоопределение колонок с текстом и с тэгом (case-insensitive, ищем подстроки)
text_col = None
tag_col = None
for c in df.columns:
    lc = c.lower()
    if 'text' in lc and text_col is None:
        text_col = c
    if ('tag' in lc or 'label' in lc) and tag_col is None:
        tag_col = c

# fallback если у вас колонки названы буквально 'TEXT_COL' и 'TAG_COL'
if text_col is None and 'TEXT_COL' in df.columns:
    text_col = 'TEXT_COL'
if tag_col is None and 'TAG_COL' in df.columns:
    tag_col = 'TAG_COL'

if text_col is None or tag_col is None:
    raise KeyError("Не удалось найти колонки с текстом/тэгами автоматически. Посмотрите вывод списка колонок выше и назначьте TEXT_COL/TAG_COL вручную.")

# Переименуем к единому виду, чтобы дальше ваш код не меняться
df = df[[text_col, tag_col]].dropna(subset=[text_col, tag_col]).reset_index(drop=True)
df = df.rename(columns={text_col: 'text', tag_col: 'tag'})

# Теперь используйте 'text' и 'tag' в дальнейшем (ваш существующий код)
TEXT_COL = 'text'
TAG_COL = 'tag'

print("Using columns:", TEXT_COL, TAG_COL)

#df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df = df[[TEXT_COL, TAG_COL]].dropna(subset=[TEXT_COL, TAG_COL]).reset_index(drop=True)
df[TEXT_COL] = df[TEXT_COL].astype(str).map(clean_text) # Датафрейм небольшой -> разница между .str и .map() будет не видна

texts = df[TEXT_COL].tolist()
tags = df[TAG_COL].astype(str).tolist()

# Кодирование тэгов
encoder = LabelEncoder()
labels = encoder.fit_transform(tags)
class_names = encoder.classes_.tolist()
num_labels = len(class_names)
print("Classes:", class_names)

# ------------------------------------------------------------------

# Применяем взвешивание, если установили USE_CLASS_WEIGHTS = True
if USE_CLASS_WEIGHTS:
    class_weights_np = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)
    print("Class weights:", class_weights_np)
else:
    class_weights = None

# ------------------------------------------------------------------

# Разделение данных
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# Токенизация
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(
    train_texts,
    truncation=True, # Усечение, если длиннее MAX_LEN
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)
val_enc = tokenizer(
    val_texts,
    truncation=True, # Усечение, если длиннее MAX_LEN
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)

# Dataset
class TagDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    # В item он уже делает вывод типа 
    # return {
    #        'input_ids': encoding['input_ids'].flatten(),
    #        'attention_mask': encoding['attention_mask'].flatten(),
    #        'labels': torch.tensor(self.labels[idx], dtype=torch.long)
    #    }

train_dataset = TagDataset(train_enc, train_labels)
val_dataset = TagDataset(val_enc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Модель
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
).to(DEVICE)

# Сохранение конфига id и тэга для дальнейшего деплоя
id2label = {str(i): c for i, c in enumerate(class_names)}
label2id = {c: int(i) for i, c in enumerate(class_names)}
model.config.id2label = id2label
model.config.label2id = label2id

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

# ------------------------------------------------------------------

# Компенсирует дисбаланс классов, если USE_CLASS_WEIGHTS = True
if class_weights is not None:
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = None

# ------------------------------------------------------------------

# Тренировка модели с AMP

# GradScaler масштабирует градиенты, чтобы предотвратить их “исчезновение”
scaler = torch.amp.GradScaler(device='cuda', enabled=torch.cuda.is_available())

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}")
    total_loss = 0.0
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        # autocast автоматически выбирает тип данных для каждой операции (где нужно float16, float32)
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            if loss_fn is None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                logits = outputs.logits
                loss = loss_fn(logits, labels_batch)

        scaler.scale(loss).backward() # Loss scaling увеличивает значения loss перед backward pass, чтобы сохранить точность
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n+1):.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

    # Оценка модели
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            true.extend(labels_batch.cpu().numpy().tolist())

    accuracy = accuracy_score(true, preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(classification_report(true, preds, target_names=class_names, digits=4))

# Сохранение модели и encoder'а для инференса и деплоя
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
print(f"Saved model, tokenizer and encoder to {OUTPUT_DIR}")

# Сохранение метаданных обучения модели
meta = {
    "seed": SEED,
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "class_names": class_names,
    "torch_version": torch.__version__
}
with open(os.path.join(OUTPUT_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)