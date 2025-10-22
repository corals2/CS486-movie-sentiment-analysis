import requests
import os
import pandas as pd
import string, time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

Key = os.getenv('TMDB_API_KEY')
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"


datafrm = pd.read_json(url)
datafrm = pd.DataFrame(datafrm)
df_r = pd.json_normalize(datafrm['results'])
moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']]
# print(moviedata.head())

#Lowercasing
overviews_title_df = pd.DataFrame()
overviews_title_df['title'] = moviedata['title']  # Keep original titles
overviews_title_df['overview'] = moviedata['overview']  # Copy overview column
overviews_title_df['overview'] = moviedata['overview'].str.lower()

#Punctuation Removal
overviews_title_df['overview'] = overviews_title_df['overview'].str.translate(str.maketrans("","",string.punctuation))

#Word Tokenization
overviews_title_df['overview'] = overviews_title_df['overview'].apply(word_tokenize)

#Stop Word Removal
overviews_title_df['overview'] = overviews_title_df['overview'].apply(lambda x: [word for word in x if word not in stopwords.words('english')])

#Stemming
port_stem = PorterStemmer()
overviews_title_df['overview'] = overviews_title_df['overview'].apply(lambda x: [port_stem.stem(word) for word in x])

print(overviews_title_df.head())
print(overviews_title_df[['overview', 'title']].head())