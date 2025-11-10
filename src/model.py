import os
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from utils import clean_text, get_grad_scaler_and_autocast
from dataset import StreamingTextDataset
import shutil
import math
import multiprocessing
import torch.nn as nn

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
        self.test_loader = None
        self.val_loader = None
        self.best_val_acc = -1.0

    def prepare_data(self, df: pd.DataFrame):
        # Автоопределение колонок
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        tag_col = next((c for c in df.columns if "tag" in c.lower() or "label" in c.lower()), None)
        if text_col is None or tag_col is None:
            raise KeyError("Не найден text/tag столбец")

        df = df[[text_col, tag_col]].dropna().reset_index(drop=True)
        df = df.rename(columns={text_col: "text", tag_col: "tag"})
        df["text"] = df["text"].map(clean_text)

        # --- Ограничить количество строк до 5000 ---
        #max_rows = 5000
        #if len(df) > max_rows:
        #    df = df.sample(n=max_rows, random_state=self.config.SEED).reset_index(drop=True)
        #    print(f"Dataset sampled to {max_rows} rows (random sample).")
        #else:
        #    print(f"Dataset size {len(df)} rows <= {max_rows}, sampling not applied.")

        texts = df["text"].tolist()
        tags = df["tag"].astype(str).tolist()

        self.encoder = LabelEncoder()
        labels = self.encoder.fit_transform(tags)
        self.class_names = self.encoder.classes_.tolist()
        self.num_labels = len(self.class_names)
        
        if self.config.USE_CLASS_WEIGHTS:
            weights = compute_class_weight(class_weight='balanced', classes=np.arange(self.num_labels), y=labels)
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.config.DEVICE)
        else:
            self.class_weights = None

        # split: train 80%, val 10%, test 10%
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=self.config.SEED,
            stratify=labels,
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=self.config.SEED,
            stratify=temp_labels,
        )

        train_dataset = StreamingTextDataset(train_texts, train_labels, self.tokenizer, max_len=self.config.MAX_LEN)
        val_dataset = StreamingTextDataset(val_texts, val_labels, self.tokenizer, max_len=self.config.MAX_LEN)
        test_dataset = StreamingTextDataset(test_texts, test_labels, self.tokenizer, max_len=self.config.MAX_LEN)

        num_workers = max(0, min(self.config.NUM_WORKERS, multiprocessing.cpu_count() - 1))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=2,
            collate_fn=self.data_collator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=self.data_collator,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=self.data_collator,
        )

        print(f"Prepared data: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME, num_labels=self.num_labels
        ).to(self.config.DEVICE)
        
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.LR)
        
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / float(self.config.GRADIENT_ACCUMULATION_STEPS))
        total_steps = num_update_steps_per_epoch * self.config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps,
        )
        
        if self.config.USE_CLASS_WEIGHTS and getattr(self, "class_weights", None) is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def save_checkpoint(self, path, epoch, is_best=False, scaler=None):
        os.makedirs(path, exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "class_names": self.class_names
        }
        fname = os.path.join(path, f"checkpoint_epoch{epoch}.pt")
        torch.save(state, fname)
        # save tokenizer + encoder separately
        self.model.save_pretrained(path + "/model_epoch{}".format(epoch))
        self.tokenizer.save_pretrained(path + "/tokenizer_epoch{}".format(epoch))
        joblib.dump(self.encoder, os.path.join(path, f"label_encoder_epoch{epoch}.joblib"))
        if is_best:
            shutil.copy(fname, os.path.join(path, "best_model.pt"))
            # and save best artifacts for quick loading
            self.model.save_pretrained(path + "/best_model")
            self.tokenizer.save_pretrained(path + "/best_tokenizer")
            joblib.dump(self.encoder, os.path.join(path, "label_encoder.joblib"))

    def load_checkpoint(self, checkpoint_path, load_optimizer=False, device=None):
        device = device or self.config.DEVICE
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        if load_optimizer and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_optimizer and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")

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
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels_batch) / float(self.config.GRADIENT_ACCUMULATION_STEPS)

                scaler.scale(loss).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(self.train_loader):
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    # step scheduler при реальном optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * float(self.config.GRADIENT_ACCUMULATION_STEPS)
                avg_loss = total_loss / (step + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")

            # evaluate on validation
            val_acc = self.evaluate(return_accuracy=True)

            # checkpointing
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(self.config.OUTPUT_DIR, epoch+1, is_best=is_best, scaler=scaler)

    def evaluate(self, loader=None, return_accuracy=False):
        """
        Evaluate model on provided loader (default: self.val_loader).
        Returns:
        - if return_accuracy=True: returns float accuracy (as before)
        - otherwise: prints and returns dict with metrics and report,
            and also returns (true, preds, probs) if you need them downstream.
        """
        self.model.eval()
        preds, true = [], []
        probs_list = []
        loader = loader or self.val_loader

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation" if loader is self.val_loader else "Evaluate"):
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels_batch = batch["labels"].to(self.config.DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape: (B, C)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                batch_preds = probs.argmax(axis=1)

                preds.extend(batch_preds.tolist())
                true.extend(labels_batch.cpu().numpy().tolist())
                probs_list.append(probs)

        # stack probs
        probs_arr = np.vstack(probs_list) if len(probs_list) > 0 else np.zeros((0, self.num_labels))
        preds = np.array(preds)
        true = np.array(true)

        # basic accuracy and sklearn report
        accuracy = float(accuracy_score(true, preds))
        report = classification_report(true, preds, target_names=self.class_names, digits=4, output_dict=True)

        # F1 scores: micro, macro, weighted
        f1_micro = float(f1_score(true, preds, average="micro"))
        f1_macro = float(f1_score(true, preds, average="macro"))
        f1_weighted = float(f1_score(true, preds, average="weighted"))

        # ROC AUC (one-vs-rest macro) — requires probs and binarized true labels
        roc_auc_macro = None
        try:
            if probs_arr.shape[0] == 0:
                roc_auc_macro = None
            else:
                n_classes = self.num_labels
                if n_classes == 2:
                    # binary: use probability of positive class (col 1)
                    roc_auc_macro = float(roc_auc_score(true, probs_arr[:, 1]))
                else:
                    # multiclass: binarize and use ovR macro
                    y_bin = label_binarize(true, classes=list(range(n_classes)))
                    roc_auc_macro = float(roc_auc_score(y_bin, probs_arr, average="macro", multi_class="ovr"))
        except Exception as e:
            # если что-то идёт не так (например, отсутствуют примеры для какого-то класса), оставляем None
            print("Warning: ROC AUC could not be computed:", str(e))
            roc_auc_macro = None

        # печать краткой сводки
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"F1 (micro/macro/weighted): {f1_micro:.4f} / {f1_macro:.4f} / {f1_weighted:.4f}")
        if roc_auc_macro is not None:
            print(f"ROC AUC (macro, OvR): {roc_auc_macro:.4f}")
        else:
            print("ROC AUC (macro, OvR): not available")

        # возвращаем структуру для дальнейшей обработки
        metrics = {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc_macro": roc_auc_macro,
            "classification_report": report,
        }

        if return_accuracy:
            return accuracy

        # возвращаем также true/preds/probs для возможного дальнейшего анализа
        return metrics, true, preds, probs_arr


    def save(self):
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.model.save_pretrained(self.config.OUTPUT_DIR + "/final_model")
        self.tokenizer.save_pretrained(self.config.OUTPUT_DIR + "/final_tokenizer")
        joblib.dump(self.encoder, os.path.join(self.config.OUTPUT_DIR, "label_encoder.joblib"))
