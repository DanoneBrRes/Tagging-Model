import torch
from torch.utils.data import Dataset

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
        return {
            "input_ids": enc['input_ids'],
            "attention_mask": enc.get('attention_mask', [1]*len(enc['input_ids'])),
            "labels": label
        }