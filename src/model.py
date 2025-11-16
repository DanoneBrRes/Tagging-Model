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
from utils import find_col_by_keywords

class TextClassifier:
    '''
    Объект подготовки данных, создания, обучения и валидации модели тэгирования.
    
    Класс оборачивает ML-пайплайн классификации текста на основе Hugging Face Transformers:
    очистка и разбиение данных, кодирование текстов и тэгов, создание DataLoader'ов, построение модели, 
    обучение с градиентным накоплением и AMP-оптимизатором, сохранение и оценка промежуточных версий, оценка качества итоговой лучшей модели.
    
    Параметры:
    ---------
    config : object
        Заранее заданные параметры для модели. 
        (device, seed, model_name, batch_size, gradient_accumulation_steps, epochs, max_len, lr, num_workers, pin_memory, use_class_weights, output_dir)
        
    Аргументы:
    ---------
    tokenizer : transformers.PreTrainedTokenizer
        tokenizer (text, truncation=True, max_length=self.max_len, padding=False, return_tensors=None)
        Возвращает mapping с ключом 'input_ids', а также 'attention_mask'. Совместим с Hugging Face API.
        Инициализируется на основе config.MODEL_NAME
    data_collector : DataCollatorWithPadding 
        Объект для работы с батчами, совместимый с HuggingFace.
    model : AutoModelForSequenceClassification
        Используемая модели.
    optimizer : torch.optim.Optimizer
        Оптимизатор AdamW при работе модели.
    scheduler : object
        Управляет LR при обучении в каждом новом шаге.
    encoder : joblib-объект
        Энкодер входящих меток в индексы.
    train_loader : torch.utils.data.DataLoader
        Загрузчик данных в обучающую выборку.
    test_loader : torch.utils.data.DataLoader
        Загрузчик данных в тестовую выборку.
    val_loader : torch.utils.data.DataLoader
        Загрузчик данных в валидационную выборку.
    best_val_acc : float
        Хранит лучшее значение точности модели, нужна как маячок для сохранения лучшей модели.
    '''
    
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
        '''
        Подготовка данных к работе: загрузка, очистка, нормализация, кодирование, разбиение на выборки и создание DataLoader'ов для каждой выборки
        
        - Автоматически ищет колонки в датафрейме по списку возможных названий колонок.
        - Очищает и нормализует текст с помощью utils.clean_text.
        - Кодирует метки LabelEncoder'ом и вычисляет class_weights, если взвешивание включено.
        - Сплитит данные на train/test/val (80/10/10) с сохранением стратификации.
        - Создает StreamingTextDataset'ы и DataLoader'ы для каждой выборки.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Входящий Датафрейм. Ожидается, что в нём есть колонка с текстами и колонка с тэгами.
            Алгоритм попытается автоматически определить имена колонок по ключевым словам в списках
            Если автоопределение не сработает, будет выброшено исключение.
        
        Исключения:
        ----------
            KeyError
                Если не найдено ни одной подходящей колонки с текстом или тэгом.
            ValueError
                Если после фильтрации или очистки не осталось ни одной строки.
            sklearn.exceptions.ValueError
                Может быть выброшен из выборок, если stratify не может быть применен (слишком мало примеров в классе).
        '''
        
        # Автоопределение колонок в датафрейме
        columns = list(df.columns)
        text_col = find_col_by_keywords(columns, ["text", "review", "comment", "body", "message", "article", "headline", "combined_text"])
        tag_col = find_col_by_keywords(columns, ["tag", "label", "category", "class", "target", "topic"])
        if text_col is None or tag_col is None:
            raise KeyError("Не найден text/tag столбец")
        
        df = df[[text_col, tag_col]].dropna().reset_index(drop=True)
        df = df.rename(columns={text_col: "text", tag_col: "tag"})
        df["text"] = df["text"].map(clean_text) # в данном случае не вижу смысла переводить текст в str, по скорости разницы не будет в обработке, у нас не так много строк

        # Ограничитель входных строк (для проверок)
        #max_rows = 500
        #if len(df) > max_rows:
        #    df = df.sample(n=max_rows, random_state=self.config.SEED).reset_index(drop=True)
        #    print(f"Датасет включает {max_rows} рандомных строк.")
        #else:
        #    print(f"Размер датасета {len(df)} строк <= {max_rows}.")

        texts = df["text"].tolist()
        tags = df["tag"].astype(str).tolist()

        # Создаем LabelEncoder для кодирования тэгов в индексы
        self.encoder = LabelEncoder()
        labels = self.encoder.fit_transform(tags)
        
        # Создается для дальнейшего декодирования тэгов из индексов в слова так, как записано в энкодере
        self.class_names = self.encoder.classes_.tolist()
        
        # Создаем переменную, отвечающую за количество уникальных значений тэгов
        # Нужна для определения размерности входящего датафрейма при обучении модели
        self.num_labels = len(self.class_names)
        
        # Блок для работы со взвешивание
        if self.config.USE_CLASS_WEIGHTS:
            weights = compute_class_weight(class_weight='balanced', classes=np.arange(self.num_labels), y=labels)
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.config.DEVICE)
        else:
            self.class_weights = None

        # Разбиение данных: train 80%, val 10%, test 10% со стратификацией
        
        # Сначала разбиваем данные как 80% в train и в 20% в temp
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=self.config.SEED,
            stratify=labels,
        )
        # затем temp выборку разбиваем пополам на val и test выборки
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=self.config.SEED,
            stratify=temp_labels,
        )

        # Создаем отдельные StreamingTextDataset для кодирования "на лету" (экономия RAM) -> только для обучения!
        train_dataset = StreamingTextDataset(train_texts, train_labels, self.tokenizer, max_len=self.config.MAX_LEN)
        val_dataset = StreamingTextDataset(val_texts, val_labels, self.tokenizer, max_len=self.config.MAX_LEN)
        test_dataset = StreamingTextDataset(test_texts, test_labels, self.tokenizer, max_len=self.config.MAX_LEN)

        # Автоматическое определение количества рабочих процессов
        num_workers = max(0, min(self.config.NUM_WORKERS, multiprocessing.cpu_count() - 1))
        
        # Создаем DataLoader'ы для батчинга и передаем в него ранее созданные датасеты
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
            num_workers=0, # для уменьшения нагрузки при оценке
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=self.data_collator,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0, # для уменьшения нагрузки при тесте
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=self.data_collator,
        )

        print(f"Подготовленные данные: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    def build_model(self):
        '''
        Построение модели: загрузка модели, оптимизатора, scheduler и loss-критерия
        
        - Загружает модель AutoModelForSequenceClassification и передает на устройство
        - Включает gradient_checkpointing (если есть): сохраняет часть активаций батчей для экономии GPU, остальные пересчитывает при backward-pass
        - Создает AdamW оптимизатор
        - Рассчитывает общее число шагов обучения (длина тренировочного датасета, деленная на градиентный накопленный шаг)
        - Инициализирует линейный scheduler на основе warmup (5% от общего количества шагов)
        - Устанавливает loss-критерий
        
        Требования:
        ----------
            Вызвать строго после prepare_data(), так как метод использует train_loader и num_labels
        '''
        
        # Загружаем предобученную модель и передаем на устройство
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME, num_labels=self.num_labels).to(self.config.DEVICE)
        
        # Включаем gradient_checkpointing
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        
        # Создаем оптимизатор AdamW
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.LR)
        
        # Вычисляем количество шагов с GRADIENT_ACCUMULATION_STEPS 
        # (условно, если у нас 100 батчей, а GRADIENT_ACCUMULATION_STEPS == 4 -> кол-во шагов будет 25)
        num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / float(self.config.GRADIENT_ACCUMULATION_STEPS))
        
        # Общее количество шагов за эпоху -> нужно для scheduler
        total_steps = num_update_steps_per_epoch * self.config.EPOCHS
        
        # Создаем линейный scheduler на основе warmup
        # Он уменьшаем learning rate (от LR до 0) на протяжении всего обучения
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps,
        )
        
        # Устанавливаем loss-критерий для двух случае: обычное обучение и взвешенное обучение
        if self.config.USE_CLASS_WEIGHTS and getattr(self, "class_weights", None) is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def save_checkpoint(self, path, epoch, is_best=False, scaler=None):
        '''
        Сохраняет промежуточные версии модели на каждой эпохе и ее значения
        
        Что сохраняет:
            - checkpoint_epoch_{epoch}.pt
                Содержит словарь с ключами:
                    "model_state"      — state_dict модели
                    "optimizer_state"  — state_dict оптимизатора (если есть)
                    "scheduler_state"  — state_dict scheduler'а (если есть)
                    "scaler_state"     — state_dict GradScaler (если передан)
                    "epoch"            — номер эпохи
                    "class_names"      — список классов
            - Модель Hugging Face
            - Токенизатор
            - LabelEncoder
            - Если это лучшая модель (if is_best == True) дублирует checkpoint и артефакты для быстрой загрузки лучшей модели в новые файлы
            (при каждом прогоне, если это была лучшая модель, переписывает уже существующий файлы лучшей модели)
        
        Параметры:
        ---------
        path : str
            Папка для сохранения файлов (будет создана, если нет)
        epoch : int
            Номер эпохи обучения
        is_best : bool
            Если True, сохраняет артефакты лучшей модели. Default False.
        scaler : torch.amp.GradScaler
            Объект GradScaler для AMP. Default None.
        '''
        
        # Создает папку (в конфиге 'output')
        os.makedirs(path, exist_ok=True)
        
        # Словарь с всеми параметрами и артефактами модели
        state = {
            "model_state": self.model.state_dict(),                                                     # веса модели nn.Module -> чтобы быстро восстановить обученную модель
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,     # внутренние параметры оптимизатора
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,     # сохраняет состояние LR у scheduler, чтобы продолжать обучение с того же состояния
            "scaler_state": scaler.state_dict() if scaler is not None else None,                        # сохраняет масштабирование градиентов в AMP
            "epoch": epoch,                                                                             # номер эпохи
            "class_names": self.class_names                                                             # чтобы декодировать индексы обратно в названия классов
        }
        
        # Создаем отдельную папку под каждую эпоху
        epoch_dir = os.path.join(path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Создаем отдельную папку под лучшую модель
        best_dir = os.path.join(path, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        
        # Создаем файл всей модели .pt для каждой эпохи
        file_name = os.path.join(epoch_dir, f"checkpoint.pt")
        
        # Сохраняем модель, затем можно будет загрузить с помощью torch.load(path)
        torch.save(state, os.path.join(epoch_dir, "checkpoint.pt"))
        
        # Сохраняем отдельно модель каждой эпохи
        self.model.save_pretrained(os.path.join(epoch_dir, "model"))
        
        # Сохраняем отдельно токенизатор каждой эпохи
        self.tokenizer.save_pretrained(os.path.join(epoch_dir, "tokenizer"))
        
        # Сохраняем отдельно энкодер каждой эпохи
        joblib.dump(self.encoder, os.path.join(epoch_dir, "label_encoder.joblib"))
        
        # Делаем сохранения для лучшей модели в своей папке
        if is_best:
            shutil.copy(file_name, os.path.join(best_dir, "best_model.pt"))
            self.model.save_pretrained(os.path.join(best_dir, "best_model"))
            self.tokenizer.save_pretrained(os.path.join(best_dir, "best_tokenizer"))
            joblib.dump(self.encoder, os.path.join(best_dir, "label_encoder.joblib"))

    # def load_checkpoint(self, checkpoint_path, load_optimizer=False, device=None):
    #    '''
    #    Загрузчик модели (пока не используется)
    #    '''
    #    device = device or self.config.DEVICE
    #    checkpoint = torch.load(checkpoint_path, map_location=device)
    #    self.model.load_state_dict(checkpoint["model_state"])
    #    if load_optimizer and "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
    #        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    #    if load_optimizer and "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None and self.scheduler is not None:
    #        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
    #    print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self):
        '''
        Метод обучения модели на test датасете.
        
        - Инициализирует grad scaler и контекст autocast через utils.get_grad_scaler_and_autocast.
        - Для каждой эпохи выполняет проход по train_loader:
            - переводит batch на device,
            - вычисляет logits через model(),
            - считает loss, gradient_accumulation_steps,
            - масштабирует loss, вызывает backward через scaler,
            - на каждом N шаге (или в конце эпохи) выполняет unscale, clip_grad_norm, optimizer.step, scaler.update,
            - делает scheduler.step (если есть).
        - После каждой эпохи выполняет очистку кэша и оценку модели через accuracy, сохраняет чекпоинт; обновляет self.best_val_acc.
        '''
        
        # Создаем AMP скейлеры через функцию utils.get_grad_scaler_and_autocast
        scaler, autocast_ctx = get_grad_scaler_and_autocast(self.config.DEVICE)

        # Основной цикл обучения по эпохам
        for epoch in range(self.config.EPOCHS):
            
            # Переводим модель в режим обучения
            self.model.train()
            # Инициализация loss для подсчета на на каждой эпохе
            total_loss = 0.0
            
            # Обнуляет градиенты всех параметров модели
            # Если не обнулить, то градиенты из предыдущих батчей суммируются, и шаг оптимизатора будет некорректным.
            self.optimizer.zero_grad()
            
            # Бар, показывающий состояние обучения
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Train epoch {epoch+1}/{self.config.EPOCHS}")

            # Проходим по всем батчам: step (индекс текущего батча), batch (словарь с данными)
            for step, batch in pbar:
                
                # Инициализируем переменные из батча
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels_batch = batch["labels"].to(self.config.DEVICE)

                # Запускаем AMP (для оптимизации вычислений: где можно использует тип float16, что уменьшает количество используемой памяти и скорость)
                with autocast_ctx():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels_batch) / float(self.config.GRADIENT_ACCUMULATION_STEPS)

                scaler.scale(loss).backward() # вычисляет градиенты для параметров модели с учётом масштабирования AMP, чтобы безопасно использовать float16 и ускорить обучение на GPU.

                # Проверка, нужно ли применять шаг оптимизатора
                # Логика условия в том, чтобы не обновлять веса при каждом новом батче, а накапливать их -> ускорение работы и экономия памяти
                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(self.train_loader):
                    
                    # Приводим к нормальному масштабу перед шагом оптимизатора
                    scaler.unscale_(self.optimizer)
                    
                    # Ограничивает норму градиентов
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Делаем шаг: обновляем веса
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    # Обнуляем градиенты после обновления весов
                    self.optimizer.zero_grad()
                    
                    # Корректно применяем scheduler при работе оптимизатора
                    if self.scheduler is not None:
                        self.scheduler.step()

                # Считаем loss при обучении
                total_loss += loss.item() * float(self.config.GRADIENT_ACCUMULATION_STEPS)
                avg_loss = total_loss / (step + 1)
                
                # Передаем в конце бара loss (динамически обновляется)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Очищаем кэш ГПУ при прогоне датасет (оптимизация использования памяти)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Считаем средний loss за эпоху
            epoch_avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")

            # Считаем accuracy модели из метода evaluate()
            val_acc = self.evaluate(return_accuracy=True)

            # Сохраняем модели из метода save_checkpoint()
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(self.config.OUTPUT_DIR, epoch+1, is_best=is_best, scaler=scaler)

    def evaluate(self, loader=None, return_accuracy=False):
        '''
        Оценка модели на val и test датасетах
        
        Параметры:
        ---------
        loader : torch.utils.data.DataLoader or None
            Если None, используется self.val_loader
        return_accuracy : bool
            Если True, возвращает только float accuracy. 
            Иначе возвращает metrics_dict, true_array, preds_array, probs_array
            
        Возвращает:
        ----------
            float or tuple:
                - Если return_accuracy=True: float (accuracy)
                - Иначе:
                    metrics: словарь с ключами 
                        accuracy,
                        f1_micro, 
                        f1_macro, 
                        f1_weighted, 
                        roc_auc_macro, 
                        classification_report.
                    true: массив истинных меток.
                    preds: массив предсказанных меток.
                    probs_arr: массив вероятностей размером (N, num_labels).
        '''
        
        # Переводит в модель в режим оценки
        self.model.eval()
        
        # Инициализация списков для хранения результатов
        preds, true, probs_list = [], [], []
        
        # Инициализируем DataLoader
        loader = loader or self.val_loader

        # Включаем вычисление градиентов -> только предсказываем
        with torch.no_grad():
            
            # Проходим по каждому батчу в DataLoader
            for batch in tqdm(loader, desc="Validation" if loader is self.val_loader else "Evaluate"):
                # Передаем словарь батча
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels_batch = batch["labels"].to(self.config.DEVICE)

                # Проходим по модели для батча
                # Инициализируем сырые данные предсказания
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Переводим сырые logits в вероятности (то есть с какой вероятностью тэг будет присвоен тексту)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # Получаем предсказанный класс для каждого примера в батче
                batch_preds = probs.argmax(axis=1)

                # Обновляем списки
                preds.extend(batch_preds.tolist())
                true.extend(labels_batch.cpu().numpy().tolist())
                probs_list.append(probs)

        # Создаем общие массивы
        probs_arr = np.vstack(probs_list) if len(probs_list) > 0 else np.zeros((0, self.num_labels)) # Создаем один массив с размерностью (N, num_labels)
        preds = np.array(preds)
        true = np.array(true)

        # Вычисляем точность модели и выводим репорт с помощью classification_report()
        accuracy = float(accuracy_score(true, preds))
        report = classification_report(true, preds, target_names=self.class_names, digits=4, output_dict=True)

        # F1-scores: micro, macro, weighted
        f1_micro = float(f1_score(true, preds, average="micro"))
        f1_macro = float(f1_score(true, preds, average="macro"))
        f1_weighted = float(f1_score(true, preds, average="weighted"))

        # ROC AUC
        roc_auc_macro = None
        
        try:
            # Если пустой массив -> None
            if probs_arr.shape[0] == 0:
                roc_auc_macro = None
            else:
                n_classes = self.num_labels
                if n_classes == 2:
                    # Бинарная классификация
                    # Сравниваем с истинными метками
                    roc_auc_macro = float(roc_auc_score(true, probs_arr[:, 1]))
                else:
                    # Многоклассовая классификация
                    # Для каждого класса считаем ROC AUC против всех остальных (OVR)
                    # Усредняем значения по всем классам
                    y_bin = label_binarize(true, classes=list(range(n_classes)))
                    roc_auc_macro = float(roc_auc_score(y_bin, probs_arr, average="macro", multi_class="ovr"))
        except Exception as e:
            print("Warning: ROC AUC не может быть вычислен", str(e))
            roc_auc_macro = None

        # Вывод сводки
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"F1 (micro/macro/weighted): {f1_micro:.4f} / {f1_macro:.4f} / {f1_weighted:.4f}")
        if roc_auc_macro is not None:
            print(f"ROC AUC (macro, OvR): {roc_auc_macro:.4f}")
        else:
            print("ROC AUC (macro, OvR): не доступен")

        # Инициализируем словарь с метриками
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

        # Возвращаем true/preds/probs для возможного дальнейшего анализа
        return metrics, true, preds, probs_arr

    def save(self):
        '''
        Метод сохранения финальной версии модели
        '''
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.model.save_pretrained(self.config.OUTPUT_DIR + "/final_model")
        self.tokenizer.save_pretrained(self.config.OUTPUT_DIR + "/final_tokenizer")
        joblib.dump(self.encoder, os.path.join(self.config.OUTPUT_DIR, "label_encoder.joblib")) 
