from torch.utils.data import Dataset

class StreamingTextDataset(Dataset):
    '''
    Датасет, который токенизирует тексты "на лету"
    
    То есть нам не нужно подгружать весь датасет в RAM, а мы берем каждый токен по одному.
    Это удобно для работы при обучении модели -> неплохо бьет по скорости, зато мы можем подгружать больший max_len, менять токенизаторы и конфиг
    
    Но для деплоя будет создан свой метод, который берет уже готовую разметку токенов в сохраненной модели (выше скорость при инференсе)
    
    Параметры:
    ---------
    texts : str 
        Последовательность исходных входящих строк
    labels : str
        Последовательность тэгов
    tokenizer : объект токенизации
        tokenizer (text, truncation=True, max_length=self.max_len, padding=False, return_tensors=None)
        Возвращает mapping с ключом 'input_ids', а также 'attention_mask'
    max_len : int
        Максимальная длина последовательности
        
    Возвращает:
    ----------
        dict: {"input_ids": list[int] или Tensor, "attention_mask": list[int] или Tensor, "labels": int}
    
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