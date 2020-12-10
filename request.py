# send requests with the features to the server and receive the result

# request sever for the predictions 

import requests 

url = 'http://localhost:5000/api'

r = requests.post(url,json={'exp':"trump wins karate"})

print(r.json())