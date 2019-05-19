import requests
# URL
url = 'http://localhost:5000/'

result = requests.post(url, json={'exp':1.8})
print("HOLLLL")
print(result.json())