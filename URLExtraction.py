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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import json

Key = os.getenv('TMDB_API_KEY')
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"
datafrm = pd.read_json(url)
datafrm = pd.DataFrame(datafrm)
df_r = pd.json_normalize(datafrm['results'])
#TODO: Increase the number of pages.
for i in range(2,501):
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page={i}"
    temp_datafrm = pd.read_json(url)
    temp_datafrm = pd.DataFrame(temp_datafrm)
    temp_df_r = pd.json_normalize(temp_datafrm['results'])
    # tmp_moviedata = temp_df_r[['title', 'genre_ids', 'overview', 'vote_average']]
    # print(df_r)
    # df_r = df_r.append(temp_df_r)
    df_r = pd.concat([df_r, temp_df_r], axis=0, ignore_index=True)
    # print(df_r)



# datafrm = pd.read_json(url)
# datafrm = pd.DataFrame(datafrm)
# df_r = pd.json_normalize(datafrm['results'])
moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']]


# text preprocessing on overview column
#Lowercasing
#overviews_title_df = moviedata['overview'].str.lower()
combined_text = moviedata['title'].str.lower() + " " + moviedata['overview'].str.lower()
overviews_title_df = pd.Series(combined_text) # include title and overview in text to process
#Punctuation Removal
overviews_title_df = overviews_title_df.str.translate(str.maketrans("","",string.punctuation))
#Word Tokenization
overviews_title_df = overviews_title_df.apply(word_tokenize)
#Stop Word Removal
overviews_title_df = overviews_title_df.apply(lambda x: [word for word in x if word not in stopwords.words('english')])
#Stemming
port_stem = PorterStemmer()
overviews_title_df = overviews_title_df.apply(lambda text: [port_stem.stem(word) for word in text])

overviews_title_df = pd.DataFrame(overviews_title_df, columns=['overview'])


# overviews_title_df = pd.DataFrame(overviews_title_df) original code 



# new_df = overviews_title_df.assign(title=df_r[['title']])
overviews_title_df.insert(0, "title", df_r[['title']])
#print(df_r[['genre_ids']].head())
#print([tuple(each_genre_id) for each_genre_id in df_r['genre_ids']])
#df_r['genre_ids'] = [tuple(each_genre_id) for each_genre_id in df_r['genre_ids']] # convert list to list of tuples for unhashable type
#df_r['genre_ids'] = [np.mean(genre_list) for genre_list in df_r['genre_ids']] # convert list of tuples to list of strings
df_r['genre_ids'] = [genre_list[0] if len(genre_list) > 0 else 1 for genre_list in df_r['genre_ids']] #grab first genre id only for simplification
#print(df_r['genre_ids'].head())

# Combined genres for lesser number of classes, for classification:
#
    # Action and Adventure [28, 12]
    # Animation and TV Movie [16, 10770]
    # Comedy [35]
    # Crime, Mystery and Thriller [80, 9648, 53]
    # Documentary and History [99, 36]
    # Drama and Family [18, 10751]
    # Romance and Music [10749, 10402]
    # Fantasy and Science Fiction [14, 878]
    # Horror [27]
    # War [10752]
    # Western [37]

# Combining the different genres as a single genre, by taking the first genre.
# df_r['genre_ids'] = [28 if genre == 28 or genre == 12 else (16 if genre == 16 or genre == 10770 else (80 if genre == 80 or genre == 9648 or genre == 53 else (99 if genre == 99 or genre == 36 else(18 if genre == 18 or genre == 10751 else(10749 if genre == 10749 or genre == 10402 else (14 if genre == 14 or genre == 878 else genre)))))) for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [16 if genre == 16 or genre == 10770 else genre for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [80 if genre == 80 or genre == 9648 or genre == 53 else genre for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [99 if genre == 99 or genre == 36 else genre for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [18 if genre == 18 or genre == 10751 else genre for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [10749 if genre == 10749 or genre == 10402 else genre for genre in df_r['genre_ids']]
# df_r['genre_ids'] = [14 if genre == 14 or genre == 878 else genre for genre in df_r['genre_ids']]
# accuracy : 0.324


    # Action and Adventure and War [28, 12, 10752]
    # Animation and TV Movie [16, 10770]
    # Crime, Mystery and Thriller and Horror [80, 9648, 53, 27]
    # Documentary and History [99, 36]
    # Drama and Music and Western [18, 10402, 37]
    # Romance and Comedy and Family[10749, 35, 10751]
    # Fantasy and Science Fiction [14, 878]
    # Western [37]
# accuracy 0.3875

df_r['genre_ids'] = [28 if genre == 28 or genre == 12 or genre == 10752 else (16 if genre == 16 or genre == 10770 else (80 if genre == 80 or genre == 9648 or genre == 53 or genre == 27 else (99 if genre == 99 or genre == 36 else(18 if genre == 18 or genre == 10402 else(10749 if genre == 10749 or genre == 35 or genre == 10751  else (14 if genre == 14 or genre == 878 else genre)))))) for genre in df_r['genre_ids']]


overviews_title_df.insert(2, "genre_ids", df_r['genre_ids'])

genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={Key}&language=en-US"
genre_json = requests.get(genre_url).text
genre_dict = json.loads(genre_json)
print(genre_dict)


# print(overviews_title_df)

# word to vec model
our_model = gensim.models.Word2Vec(
    window=15, # window around target word
    min_count=2,
)

# build vocab from overviews column in dataframe, this is a list of unique words
our_model.build_vocab(overviews_title_df['overview'])
# print(our_model.wv.index_to_key)

# train the word2vec model.
our_model.train(overviews_title_df['overview'], total_examples=our_model.corpus_count, epochs=10)

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
    if len(vector) == 0:
        vector.append(np.zeros(our_model.vector_size)) # if no words in overview, append zero vector
        #print("ZERO VECTOR APPENDED")
    # x.append(np.sum(vector, axis=0))
    x.append(np.sum(vector, axis=0)) # x is the list of overview vectors




#print(x) # print first overview vector

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1) # split data into training and testing sets

# #Using RandomForestClassifier to train and test the model, and then calculating the accuracy score based on the prediction.
# our_rfc = RandomForestClassifier()

# #Training the model.
# our_rfc.fit(x_train, y_train)
# accuracy = 0.382, cv accuracy = 0.3852

#Using RandomForestClassifier with hyperparameter tuning.
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=3)
our_rfc = grid.fit(x_train, y_train).best_estimator_
# accuracy = 0.451, cv accuracy = 0.4401

#Testing the model, getting the predicted label of the model.
y_prediction = our_rfc.predict(x_test)

#Calculating the accuracy score of the model, comparing the prediction with the true label.
print(accuracy_score(y_prediction, y_test))
print("Cross-validated accuracy:", cross_val_score(our_rfc, x, y, cv=5).mean())