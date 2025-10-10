import requests


url = ""+env+""
response = requests.get(url)

json_res = response.json()
