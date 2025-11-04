import numpy as np
import requests
import os
import pandas as pd
import gensim
import string, time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

Key = os.getenv('TMDB_API_KEY')
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"


datafrm = pd.read_json(url)
datafrm = pd.DataFrame(datafrm)
df_r = pd.json_normalize(datafrm['results'])
moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']]

# text preprocessing on overview column
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
#print(df_r[['genre_ids']].head())
#print([tuple(each_genre_id) for each_genre_id in df_r['genre_ids']])
#df_r['genre_ids'] = [tuple(each_genre_id) for each_genre_id in df_r['genre_ids']] # convert list to list of tuples for unhashable type
df_r['genre_ids'] = [np.mean(genre_list) for genre_list in df_r['genre_ids']] # convert list of tuples to list of strings
overviews_title_df.insert(2, "genre_ids", df_r['genre_ids'])

genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={Key}&language=en-US"
genre_json = requests.get(genre_url).text
genre_dict = json.loads(genre_json)
print(genre_dict)


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

#print(our_model.wv.most_similar("banker"))

# Label Encoder for movie classification based on genre
our_encoder = LabelEncoder()
y = our_encoder.fit_transform(overviews_title_df["genre_ids"]) # fit and transform genre ids into numerical labels


x = [] # list to hold overview vectors
for overview in overviews_title_df['overview']: # create vectors for each overview
    vector = []
    for word in overview:
        if word in our_model.wv:
            vector.append(our_model.wv[word])
    #print(f"Overview: {overview}")
    #print(f"Vector: {vector}")
    x.append(np.mean(vector, axis=0)) # x is the list of overview vectors



#print(x) # print first overview vector

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1) # split data into training and testing sets

#Using RandomForestClassifier to train and test the model, and then calculating the accuracy score based on the prediction.
our_rfc = RandomForestClassifier()

#Training the model.
our_rfc.fit(x_train, y_train)

#Testing the model, getting the predicted label of the model.
y_prediction = our_rfc.predict(x_test)

#Calculating the accuracy score of the model, comparing the prediction with the true label.
print(accuracy_score(y_prediction, y_test))