import random
import re
import torch
import numpy as np

def set_seed(seed: int):
    '''
    Метод для создания рандомного сида для модели
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text: str) -> str:
    '''
    Метод очистки текстов в датафрейме
    
    Возвращает:
    -----------
        text : str
            Очищенный текст
    '''
    
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r'[^\w\s]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_grad_scaler_and_autocast(device):
    '''
    Метод для инициализации AMP в обучении
    
    Возвращает:
    ----------
        scaler : torch.amp.GradScaler
        autocast_ctx : Callable
    '''
    
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

def find_col_by_keywords(columns, keywords):
    '''
    Метод для автоматического определения колонок в датафрейме
        
    Параметры:
    -----------
        columns : list of str
            Список названий колонок датафрейма
        keywords : list of str
            Список ключевых слов для поиска
        
    Возвращает:
    -----------
        c (column) : str
        
    Пример:
    --------
    find_col_by_keywords(['text', 'label'], ['label'])
    'label'
    '''
    
    lc = [c.lower() for c in columns]
    for kw in keywords:
        for c, cl in zip(columns, lc):
            if kw in cl:
                return c
    return None
