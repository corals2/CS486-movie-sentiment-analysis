import requests
import os
import pandas as pd

Key = os.getenv('TMDB_API_KEY')
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"


datafrm = pd.read_json(url)
datafrm = pd.DataFrame(datafrm)
df_r = pd.json_normalize(datafrm['results'])
moviedata = df_r[['title', 'genre_ids', 'overview', 'vote_average']]
print(moviedata.head())

