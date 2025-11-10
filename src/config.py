import torch

class Config:
    '''
    Конфигурационный класс для хранения гиперпараметров модели.
    
    Хранит выбор устройства (device), seed и основные гиперпараметры обучения.
    
    Объект только сохраняет значения.
    Он не хранит глобальное состояние. Для этого впоследствии будет добавлен метод.
    
    Параметры:
    ---------
    device : None | str | torch.device
        Устройство для обучения модели. Если None -> выбирается CUDA (если есть), иначе CPU
    seed : int
        Рандом сид для модели. Default == 42
    model_name : str
        Название используемой модели. Default = 'cointegrated/rubert-tiny2'
    batch_size : int
        Размер батча на шаг. Default == 4
    gradient_accumulation_steps : int
        Количество шагов накопления градиента. Default == 8
    epochs : int
        Количество эпох обучения. Default == 8
    max_len : int
        Максимальная длина токенов. Default == 128
    lr : float
        Скорость обучения (learning rate). Default == 2e-5
    num_workers : int
        Количество рабочих асинхронных процессов Dataloader. Default == 2
    pin_memory : None | bool
        Если None, устанавливается в True при наличии CUDA
    use_class_weights : bool
        Параметр для взвешивания переменных. Default == True
    output_dir : str
        Каталог вывода
    '''
    def __init__(
        self,
        device=None,
        seed=42,
        model_name="cointegrated/rubert-tiny2",
        batch_size=4,
        gradient_accumulation_steps=8,
        epochs=8,
        max_len=128,
        lr=2e-5,
        num_workers=2,
        pin_memory=None,
        use_class_weights=True,
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
