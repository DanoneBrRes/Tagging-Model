import os
import numpy as np
import torch
from utils import clean_text
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

class InferenceService():
    def __init__(self, model, tokenizer, label_encoder, device = None, max_length = 128, batch_size = 4, path = 'output/'):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.batch_size = batch_size
        self.path = path
        self.model.to(self.device)
        self.model.eval()
        self.warmup_done = False
        self.model_version = getattr(model.config, 'model_version', None) or model.config.to_dict()
        
    @classmethod 
    def from_pretrained(cls, path, device = None, **kwargs):
        
        model_dir = os.path.join(path, 'best_model')
        tokenizer_dir = os.path.join(path, 'best_tokenizer')
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        label_encoder = joblib.load(path + '/label_encoder.joblib')
        return cls(model, tokenizer, label_encoder, device=device, **kwargs)
    
    def preprocess(self, text: str) -> str:
        text_clean = text(clean_text)
        if text_clean is None:
            return ''
        return text_clean.strip()
    
    def _tokenize(self, texts: List[str], padding='longest') -> Dict[str, torch.Tensor]:
        texts = ['' if t in texts is None else str(t).strip() for t in texts]
        if len(texts) == 0:
            texts = []
        enc = self.tokenizer(texts, truncation = True, padding=padding, max_length = self.max_length, return_tensors = 'pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc
    
    def _forward(self, batch_tensors) -> np.ndarray:
        pass
    
    def _postprocess(self, probs: np.ndarray, top_k=1, threshold=None) -> None:
        pass
    
    def predict(self):
        pass
    
    def predict_single(self):
        pass
    
    def warmup(self):
        pass
    
    