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
import multiprocessing

# --- Конфигурация ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MODEL_NAME = "cointegrated/rubert-tiny2"
BATCH_SIZE = 8
EPOCHS = 4
MAX_LEN = 256
LR = 2e-5
NUM_WORKERS = 4  # на Windows можно временно поставить 0 для отладки
USE_CLASS_WEIGHTS = False
OUTPUT_DIR = "./output"  # обязательно не пустая строка

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_grad_scaler_and_autocast():
    use_amp = torch.cuda.is_available()
    # try new unified API, fallback to older cuda.amp
    try:
        scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=use_amp)
        def autocast_ctx():
            return torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)
        def autocast_ctx():
            return torch.amp.autocast(enabled=use_amp)
    return scaler, autocast_ctx

# --- Ваши функции очистки / датасет и т.д. ---
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r'[^\w\s]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class TagDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def main():
    set_seed()

    # чтение CSV + автопоиск колонок (как у вас)
    df = pd.read_csv('combined_news_dataset_V2.csv', sep=',')
    # ... автоопределение text/tag как раньше ...
    text_col = None
    tag_col = None
    for c in df.columns:
        lc = c.lower()
        if 'text' in lc and text_col is None:
            text_col = c
        if ('tag' in lc or 'label' in lc) and tag_col is None:
            tag_col = c
    if text_col is None and 'TEXT_COL' in df.columns:
        text_col = 'TEXT_COL'
    if tag_col is None and 'TAG_COL' in df.columns:
        tag_col = 'TAG_COL'
    if text_col is None or tag_col is None:
        raise KeyError("Не найден text/tag столбец. Посмотрите df.columns.")

    df = df[[text_col, tag_col]].dropna().reset_index(drop=True)
    df = df.rename(columns={text_col: 'text', tag_col: 'tag'})
    df['text'] = df['text'].astype(str).map(clean_text)

    # --- Ограничить количество строк до 5000 ---
    max_rows = 50000
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
        print(f"Dataset sampled to {max_rows} rows (random sample).")
    else:
        print(f"Dataset size {len(df)} rows <= {max_rows}, sampling not applied.")
    # -------------------------------------------

    texts = df['text'].tolist()
    tags = df['tag'].astype(str).tolist()

    encoder = LabelEncoder()
    labels = encoder.fit_transform(tags)
    class_names = encoder.classes_.tolist()
    num_labels = len(class_names)

    # class weights
    if USE_CLASS_WEIGHTS:
        class_weights_np = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)
    else:
        class_weights = None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

    train_dataset = TagDataset(train_enc, train_labels)
    val_dataset = TagDataset(val_enc, val_labels)

    # pin_memory только если CUDA есть
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else None

    # GradScaler + autocast (кросс-версия)
    scaler, autocast_ctx = get_grad_scaler_and_autocast()

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["labels"].to(DEVICE)

            with autocast_ctx():
                if loss_fn is None:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    loss = outputs.loss
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels_batch)

            scaler.scale(loss).backward()
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

    # сохранение (не забудьте OUTPUT_DIR не пуст)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

if __name__ == "__main__":
    # На Windows полезно явно вызвать freeze_support
    multiprocessing.freeze_support()
    # Опционально можно явно установить стартовый метод:
    # multiprocessing.set_start_method('spawn', force=True)
    main()
