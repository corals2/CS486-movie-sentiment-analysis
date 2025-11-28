import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BERT-Based Movie Recommendation System")
print("="*60)

# ============================================
# STEP 1: LOAD TRAINED MODEL
# ============================================
print("\nLoading trained BERT model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = './bert_movie_genre_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

print("Model loaded successfully!\n")

# ============================================
# STEP 2: LOAD MOVIE DATABASE
# ============================================
print("Loading movie database...")
import os
import requests

Key = os.getenv('TMDB_API_KEY')
if not Key:
    raise ValueError("TMDB_API_KEY environment variable not set!")

# Fetch movie data (same as training)
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"
datafrm = pd.read_json(url)
df_r = pd.json_normalize(datafrm['results'])

# Fetch multiple pages
from tqdm import tqdm
for i in tqdm(range(2, 101), desc="Fetching movie database"):  # Reduced to 100 pages for faster loading
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page={i}"
    try:
        temp_datafrm = pd.read_json(url)
        temp_df_r = pd.json_normalize(temp_datafrm['results'])
        df_r = pd.concat([df_r, temp_df_r], axis=0, ignore_index=True)
    except:
        continue

moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']].copy()
moviedata = moviedata.dropna(subset=['overview'])
moviedata = moviedata[moviedata['overview'].str.strip() != '']
moviedata['text'] = moviedata['title'] + " [SEP] " + moviedata['overview']

print(f"Loaded {len(moviedata)} movies\n")

# ============================================
# STEP 3: EXTRACT BERT EMBEDDINGS FOR ALL MOVIES
# ============================================
print("Extracting BERT embeddings for all movies...")

def get_bert_embedding(text, model, tokenizer, device):
    """Extract BERT embedding (CLS token) for a given text"""
    encoding = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding

# Extract embeddings for all movies
embeddings = []
for text in tqdm(moviedata['text'].tolist(), desc="Computing embeddings"):
    emb = get_bert_embedding(text, model, tokenizer, device)
    embeddings.append(emb[0])

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}\n")

# ============================================
# STEP 4: PREDICTION FUNCTIONS
# ============================================

def predict_genre(movie_title_or_description, top_k=1):
    """Predict genre for a given movie title or description"""
    encoding = tokenizer(movie_title_or_description, truncation=True, padding=True, max_length=256, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
    
    results = []
    for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
        results.append({
            'genre_id': model.config.id2label[idx] if hasattr(model.config, 'id2label') else idx,
            'probability': prob
        })
    
    return results


def recommend_similar_movies(movie_title, top_n=10):
    """Recommend similar movies based on BERT embeddings"""
    # Find the movie in database
    matches = moviedata[moviedata['title'].str.contains(movie_title, case=False, na=False)]
    
    if len(matches) == 0:
        print(f"Movie '{movie_title}' not found in database.")
        print("Try searching with partial title or use custom description.\n")
        return None
    
    # Use first match
    movie_idx = matches.index[0]
    movie_embedding = embeddings[movie_idx].reshape(1, -1)
    
    # Calculate cosine similarity with all movies
    similarities = cosine_similarity(movie_embedding, embeddings)[0]
    
    # Get top N similar movies (excluding the movie itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            'title': moviedata.iloc[idx]['title'],
            'similarity': similarities[idx],
            'overview': moviedata.iloc[idx]['overview'][:150] + "...",
            'rating': moviedata.iloc[idx]['vote_average']
        })
    
    return recommendations


def recommend_by_description(description, top_n=10):
    """Recommend movies based on a custom description"""
    # Get embedding for the description
    desc_embedding = get_bert_embedding(description, model, tokenizer, device).reshape(1, -1)
    
    # Calculate cosine similarity with all movies
    similarities = cosine_similarity(desc_embedding, embeddings)[0]
    
    # Get top N similar movies
    similar_indices = np.argsort(similarities)[::-1][:top_n]
    
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            'title': moviedata.iloc[idx]['title'],
            'similarity': similarities[idx],
            'overview': moviedata.iloc[idx]['overview'][:150] + "...",
            'rating': moviedata.iloc[idx]['vote_average']
        })
    
    return recommendations


# ============================================
# STEP 5: INTERACTIVE DEMO
# ============================================

def display_recommendations(recommendations):
    """Pretty print recommendations"""
    if recommendations is None:
        return
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Similarity: {rec['similarity']:.4f} | Rating: {rec['rating']}/10")
        print(f"   {rec['overview']}")
    print("="*60 + "\n")


def main():
    """Main interactive loop"""
    print("\n" + "="*60)
    print("MOVIE RECOMMENDATION OPTIONS")
    print("="*60)
    print("1. Find similar movies by title")
    print("2. Find movies by custom description")
    print("3. Predict genre for a movie/description")
    print("4. Exit")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            movie_title = input("Enter movie title (or partial title): ").strip()
            top_n = input("How many recommendations? (default 10): ").strip()
            top_n = int(top_n) if top_n else 10
            
            print(f"\nSearching for movies similar to '{movie_title}'...")
            recommendations = recommend_similar_movies(movie_title, top_n)
            display_recommendations(recommendations)
        
        elif choice == '2':
            description = input("Enter movie description: ").strip()
            top_n = input("How many recommendations? (default 10): ").strip()
            top_n = int(top_n) if top_n else 10
            
            print(f"\nFinding movies matching your description...")
            recommendations = recommend_by_description(description, top_n)
            display_recommendations(recommendations)
        
        elif choice == '3':
            text = input("Enter movie title or description: ").strip()
            print(f"\nPredicting genre...")
            genres = predict_genre(text, top_k=3)
            print("\nTop 3 Genre Predictions:")
            for i, g in enumerate(genres, 1):
                print(f"{i}. Genre ID: {g['genre_id']} | Probability: {g['probability']:.4f}")
            print()
        
        elif choice == '4':
            print("\nThank you for using the Movie Recommendation System!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


# ============================================
# EXAMPLE USAGE (UNCOMMENT TO TEST)
# ============================================

# Example 1: Find movies similar to "The Shawshank Redemption"
print("\n" + "="*60)
print("EXAMPLE 1: Movies similar to 'The Shawshank Redemption'")
print("="*60)
recommendations = recommend_similar_movies("Shawshank", top_n=5)
display_recommendations(recommendations)

# Example 2: Find movies by description
print("\n" + "="*60)
print("EXAMPLE 2: Movies about 'space exploration and aliens'")
print("="*60)
recommendations = recommend_by_description("space exploration and aliens", top_n=5)
display_recommendations(recommendations)

# Example 3: Predict genre
print("\n" + "="*60)
print("EXAMPLE 3: Predict genre for 'A thrilling heist in Las Vegas'")
print("="*60)
genres = predict_genre("A thrilling heist in Las Vegas", top_k=3)
print("\nTop 3 Genre Predictions:")
for i, g in enumerate(genres, 1):
    print(f"{i}. Genre ID: {g['genre_id']} | Probability: {g['probability']:.4f}")

# Start interactive mode
print("\n" + "="*60)
print("Starting Interactive Mode...")
print("="*60)
main()
