import requests

try:
    response = requests.get("http://127.0.0.1:8009/ask?question=What+are+the+capital+requirements")
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(e)
