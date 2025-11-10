import os
import random
import joblib
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing
import math

class Config:
    def __init__(
        self,
        device=None,
        seed=42,
        model_name="cointegrated/rubert-tiny2",
        batch_size=16,
        gradient_accumulation_steps=16,
        epochs=8,
        max_len=128,
        lr=2e-5,
        num_workers=8,
        pin_memory=None,
        use_class_weights=False,
        output_dir="./output"
    ):
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SEED = seed
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = batch_size
        self.GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps
        self.EPOCHS = epochs
        self.MAX_LEN = max_len
        self.LR = lr
        self.NUM_WORKERS = num_workers
        self.PIN_MEMORY = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.USE_CLASS_WEIGHTS = use_class_weights
        self.OUTPUT_DIR = output_dir
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r'[^\w\s]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_grad_scaler_and_autocast(device):
    use_amp = torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
        def autocast_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)
        def autocast_ctx():
            return torch.amp.autocast(enabled=use_amp)
    return scaler, autocast_ctx


class StreamingTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc.get("attention_mask", [1]*len(enc["input_ids"])), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }
        

class TextClassifier:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME, use_fast=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.encoder = None
        self.train_loader = None
        self.val_loader = None

    def prepare_data(self, df: pd.DataFrame):
        # Автоопределение колонок
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        tag_col = next((c for c in df.columns if "tag" in c.lower() or "label" in c.lower()), None)
        if text_col is None or tag_col is None:
            raise KeyError("Не найден text/tag столбец")

        df = df[[text_col, tag_col]].dropna().reset_index(drop=True)
        df = df.rename(columns={text_col: "text", tag_col: "tag"})
        df["text"] = df["text"].map(clean_text)

        texts = df["text"].tolist()
        tags = df["tag"].astype(str).tolist()

        self.encoder = LabelEncoder()
        labels = self.encoder.fit_transform(tags)
        self.class_names = self.encoder.classes_.tolist()
        self.num_labels = len(self.class_names)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=self.config.SEED,
            stratify=labels,
        )

        train_dataset = StreamingTextDataset(train_texts, train_labels, self.tokenizer, max_len=self.config.MAX_LEN)
        val_dataset = StreamingTextDataset(val_texts, val_labels, self.tokenizer, max_len=self.config.MAX_LEN)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=2,
            collate_fn=self.data_collator
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=self.data_collator
        )

    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME, num_labels=self.num_labels
        ).to(self.config.DEVICE)
        self.model.gradient_checkpointing_enable()
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.LR)
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.config.GRADIENT_ACCUMULATION_STEPS)
        total_steps = num_update_steps_per_epoch * self.config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps,
        )

    def train(self):
        scaler, autocast_ctx = get_grad_scaler_and_autocast(self.config.DEVICE)

        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Train epoch {epoch+1}/{self.config.EPOCHS}")

            for step, batch in pbar:
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels_batch = batch["labels"].to(self.config.DEVICE)

                with autocast_ctx():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    loss = outputs.loss / self.config.GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
                pbar.set_postfix({"loss": f"{total_loss / (step+1):.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Epoch {epoch+1} avg loss: {total_loss / len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels_batch = batch["labels"].to(self.config.DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds.tolist())
                true.extend(labels_batch.cpu().numpy().tolist())

        accuracy = accuracy_score(true, preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(true, preds, target_names=self.class_names, digits=4))

    def save(self):
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.model.save_pretrained(self.config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(self.config.OUTPUT_DIR)
        joblib.dump(self.encoder, os.path.join(self.config.OUTPUT_DIR, "label_encoder.joblib"))
        

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    df = pd.read_csv("combined_news_dataset_V2.csv", sep=",")
    classifier = TextClassifier(cfg)
    classifier.prepare_data(df)
    classifier.build_model()
    classifier.train()
    classifier.evaluate()
    classifier.save()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()