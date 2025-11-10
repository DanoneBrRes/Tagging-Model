import torch

class Config:
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
