import random
import re
import torch
import numpy as np

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
