import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW  # Исправленный импорт
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# Конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'cointegrated/rubert-tiny2'
BATCH_SIZE = 16
EPOCHS = 6
MAX_LEN = 320

# Загрузка данных
df = pd.read_excel('sampled_comments.xlsx', sheet_name='Упоминания')
comments = df['Комментарий'].tolist()
labels = df['Тональность'].tolist()

# Очистка текста
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # удалить URL
    text = re.sub(r'@\w+', '', text)     # удалить упоминания
    text = re.sub(r'[^\w\s]', ' ', text) # удалить спецсимволы
    text = re.sub(r'\s+', ' ', text)     # удалить лишние пробелы
    return text.strip()

cleaned_comments = [clean_text(str(c)) for c in comments]

# Кодирование меток
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
class_names = encoder.classes_.tolist()

# Разделение данных
train_texts, val_texts, train_labels, val_labels = train_test_split(
    cleaned_comments, encoded_labels, test_size=0.2, random_state=42
)

# Датасет
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(class_names)
).to(DEVICE)

# Создание DataLoader
train_dataset = CommentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = CommentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5)

# Обучение
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_train_loss:.4f}")

# Оценка модели
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Validation'):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")

# Сохранение модели
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')
print("Model saved!")