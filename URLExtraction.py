import requests
import os
import pandas as pd
import gensim
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

#Lowercasing
overviews_title_df = moviedata['overview'].str.lower()

#Punctuation Removal
overviews_title_df = overviews_title_df.str.translate(str.maketrans("","",string.punctuation))

#Word Tokenization
overviews_title_df = overviews_title_df.apply(word_tokenize)

#Stop Word Removal
overviews_title_df = overviews_title_df.apply(lambda x: [word for word in x if word not in stopwords.words('english')])

#Stemming
port_stem = PorterStemmer()
overviews_title_df = overviews_title_df.apply(lambda text: [port_stem.stem(word) for word in text])

overviews_title_df = pd.DataFrame(overviews_title_df)

# new_df = overviews_title_df.assign(title=df_r[['title']])
overviews_title_df.insert(0, "title", df_r[['title']])
overviews_title_df.insert(2, "genre_ids", df_r[['genre_ids']])

genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={Key}&language=en-US"


print(overviews_title_df.head())

# word to vec model
our_model = gensim.models.Word2Vec(
    window= 10, # window around target word
    min_count=1,
)

# build vocab from overviews column in dataframe, this is a list of unique words
our_model.build_vocab(overviews_title_df['overview'])
# print(our_model.wv.index_to_key)

# train the word2vec model.
our_model.train(overviews_title_df['overview'], total_examples=our_model.corpus_count, epochs=our_model.epochs)

print(our_model.wv.most_similar("banker"))

#