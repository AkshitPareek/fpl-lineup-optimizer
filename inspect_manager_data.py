import requests
import json

# Fetch directly from FPL API to see raw structure (simulating what service does)
# But better to use the service code logic if possible. 
# Let's just use the backend endpoint /api/manager/{id} if it exposes raw data?
# No, endpoint returns optimization result or list of picks? 
# The endpoint /api/manager/{manager_id} returns "picks" and "entry_history".

# Let's hit our own backend endpoint for manager data if it exists?
# I added /api/manager/{manager_id} in main.py?
# Let's check main.py

# ...
# @app.get("/api/manager/{manager_id}")
# ...

url = "http://localhost:8000/api/manager/9777842"
try:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("Keys in response:", data.keys())
        if "picks" in data:
            print("First pick sample:", data["picks"][0])
    else:
        print("Error:", response.text)
except Exception as e:
    print("Error:", e)
