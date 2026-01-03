import requests
import json
import time

url = "http://localhost:8000/api/optimize"
payload = {
    "budget": 100,
    "gameweeks": 1,
    "strategy": "transfers",
    "manager_id": 9777842,
    "free_transfers": 2
}

try:
    print(f"Sending request with {payload['free_transfers']} free transfers...")
    start_time = time.time()
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        if data.get('transfers'):
             print(f"Transfers In: {data['transfers']['in']}")
             print(f"Transfers Out: {data['transfers']['out']}")
             print(f"Cost: {data['transfers']['cost']}")
    else:
        print(response.text)
    print(f"Time: {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"Error: {e}")
