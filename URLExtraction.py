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
overviews_title_df = moviedata['overview'].str.lower()

#Punctuation Removal
overviews_title_df = overviews_title_df.str.translate(str.maketrans("","",string.punctuation))

#Word Tokenization
overviews_title_df = overviews_title_df.apply(word_tokenize)
# print(overviews_title_df[0])

#Stop Word Removal
# print(stopwords.words('english'))
overviews_title_df = overviews_title_df.apply(lambda x: [word for word in x if word not in stopwords.words('english')])

#Stemming
port_stem = PorterStemmer()
overviews_title_df = overviews_title_df.apply(lambda text: [port_stem.stem(word) for word in text])

overviews_title_df = pd.DataFrame(overviews_title_df)

# new_df = overviews_title_df.assign(title=df_r[['title']])
new_df = overviews_title_df.insert(0, "title", df_r[['title']])

genre_url = "https://api.themoviedb.org/3/genre/movie/list?api_key={Key}&language=en-US"

print(overviews_title_df[0])