import numpy as np
import requests
import os
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Starting BERT-based Movie Genre Classification...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================
# STEP 1: DATA FETCHING FROM TMDB API
# ============================================
print("Fetching data from TMDB API...")
Key = os.getenv('TMDB_API_KEY')
if not Key:
    raise ValueError("TMDB_API_KEY environment variable not set!")

url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"
datafrm = pd.read_json(url)
df_r = pd.json_normalize(datafrm['results'])

# Fetch multiple pages for more data
for i in tqdm(range(2, 501), desc="Fetching pages"):
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page={i}"
    try:
        temp_datafrm = pd.read_json(url)
        temp_df_r = pd.json_normalize(temp_datafrm['results'])
        df_r = pd.concat([df_r, temp_df_r], axis=0, ignore_index=True)
    except:
        print(f"Error fetching page {i}, skipping...")
        continue

print(f"Total movies fetched: {len(df_r)}\n")

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("Preprocessing data...")

# Extract relevant columns
moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']].copy()

# Remove rows with missing overviews
moviedata = moviedata.dropna(subset=['overview'])
moviedata = moviedata[moviedata['overview'].str.strip() != '']

# Extract first genre ID for simplification
moviedata['genre_id'] = moviedata['genre_ids'].apply(lambda x: x[0] if len(x) > 0 else None)
moviedata = moviedata.dropna(subset=['genre_id'])

# Genre grouping for better classification (same as original)
genre_mapping = {
    28: 28, 12: 28, 10752: 28,  # Action/Adventure/War
    16: 16, 10770: 16,  # Animation/TV Movie
    80: 80, 9648: 80, 53: 80, 27: 80,  # Crime/Mystery/Thriller/Horror
    99: 99, 36: 99,  # Documentary/History
    18: 18, 10402: 18,  # Drama/Music
    10749: 10749, 35: 10749, 10751: 10749,  # Romance/Comedy/Family
    14: 14, 878: 14  # Fantasy/Sci-Fi
}

moviedata['genre_id'] = moviedata['genre_id'].map(lambda x: genre_mapping.get(x, x))

# Combine title and overview for richer context
moviedata['text'] = moviedata['title'] + " [SEP] " + moviedata['overview']

# Filter out genres with very few samples (< 10)
genre_counts = moviedata['genre_id'].value_counts()
valid_genres = genre_counts[genre_counts >= 10].index
moviedata = moviedata[moviedata['genre_id'].isin(valid_genres)]

print(f"Movies after preprocessing: {len(moviedata)}")
print(f"Genre distribution:\n{moviedata['genre_id'].value_counts()}\n")

# ============================================
# STEP 3: LABEL ENCODING
# ============================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(moviedata['genre_id'])
num_classes = len(label_encoder.classes_)

print(f"Number of classes: {num_classes}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}\n")

# ============================================
# STEP 4: BERT TOKENIZATION
# ============================================
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all texts
print("Tokenizing texts...")
texts = moviedata['text'].tolist()
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')

# ============================================
# STEP 5: CREATE PYTORCH DATASET
# ============================================
class MovieDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Train-test split with stratification
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    range(len(y)), y, test_size=0.2, random_state=42, stratify=y
)

# Create train and test encodings
train_encodings = {key: val[X_train_idx] for key, val in encodings.items()}
test_encodings = {key: val[X_test_idx] for key, val in encodings.items()}

train_dataset = MovieDataset(train_encodings, y_train)
test_dataset = MovieDataset(test_encodings, y_test)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

# ============================================
# STEP 6: BERT MODEL SETUP
# ============================================
print("Loading BERT model for sequence classification...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# ============================================
# STEP 7: TRAINING SETUP
# ============================================
batch_size = 16
epochs = 4
learning_rate = 2e-5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# ============================================
# STEP 8: TRAINING LOOP
# ============================================
print("Starting training...\n")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
    # Training
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_train_loss += loss.item()
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}\n")

# ============================================
# STEP 9: EVALUATION
# ============================================
print("Evaluating on test set...\n")
model.eval()

test_predictions = []
test_true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        test_predictions.extend(predictions.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

# Calculate metrics
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')

print("="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score (weighted): {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_true_labels, test_predictions, target_names=[str(c) for c in label_encoder.classes_]))

# ============================================
# STEP 10: SAVE MODEL (OPTIONAL)
# ============================================
print("\nSaving model...")
model.save_pretrained('./bert_movie_genre_model')
tokenizer.save_pretrained('./bert_movie_genre_model')
print("Model saved to './bert_movie_genre_model'")

print("\n" + "="*50)
print("Training and evaluation complete!")
print("="*50)