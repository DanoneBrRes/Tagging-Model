from torch.utils.data import Dataset

class StreamingTextDataset(Dataset):
    '''
    Объект PyTorch Dataset для потоковой токенизации текстов ("на лету").
    
    Вместо предварительной токенизации всего фрейма, тексты обрабатываются непосредственно при обращении к элементу датасета.
    Это снижает нагрузку на оперативную память и позволяет динамически менять параметры токенизатора.
    
    Такой подход не удобен для деплоя. 
    Будет создан свой метод, который берет уже готовую разметку токенов в сохраненной модели,
    так как производительность будет выше
    
    Параметры:
    ---------
    texts : str 
        Последовательность исходных входящих строк
    labels : str
        Последовательность тэгов
    tokenizer : transformers.PreTrainedTokenizer
        tokenizer (text, truncation=True, max_length=self.max_len, padding=False, return_tensors=None)
        Возвращает mapping с ключом 'input_ids', а также 'attention_mask'. Совместим с Hugging Face API.
    max_len : int
        Максимальная длина последовательности
        
    Возвращает:
    ----------
        dict
            - "input_ids": list[int] — идентификаторы токенов;
            - "attention_mask": list[int] — маска;
            - "labels": int — целевой тэг.
    
    Примечания:
    ----------
        1. labels ожидает только одиночный тэг. Если будет мульти-тэг, он ляжет. Нужно добавить обработку multi-label случаев
    '''
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        '''Возвращает количество элементов в датасете'''
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Возвращает токенизированный элемент датасета по индексу.
        Возвращает словарь, совместимый с DataCollatorWithPadding.
        '''
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