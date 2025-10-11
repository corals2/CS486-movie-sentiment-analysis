import requests
import os
import pandas as pd

Key = os.getenv('TMDB_API_KEY')
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={Key}&language=en-US&page=1"
# response = requests.get(url)

# json_res = response.json()

# print(json_res)

datafrm = pd.read_json(url)
datafrm = pd.DataFrame(datafrm)

print(datafrm.head)