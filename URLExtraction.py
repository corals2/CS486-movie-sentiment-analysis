import requests


url = "https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}"
response = requests.get(url)

json_res = response.json()

print(json_res)