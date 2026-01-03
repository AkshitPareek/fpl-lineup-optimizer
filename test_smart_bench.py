import requests
import json

url = "http://localhost:8000/api/optimize"
payload = {
    "budget": 100,
    "gameweeks": 1,
    "strategy": "standard",
    "manager_id": None
}

try:
    print("Calling API...")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        lineup = data.get("lineup", [])
        
        starters = [p for p in lineup if p.get("is_starter")]
        bench = [p for p in lineup if not p.get("is_starter")]
        
        print(f"Total Players: {len(lineup)}")
        print(f"Starters: {len(starters)}")
        print(f"Bench: {len(bench)}")
        
        if len(starters) == 11 and len(bench) == 4:
            print("SUCCESS: 11 starters and 4 bench players found.")
            
            # Check positions
            print("Starters by Position:")
            positions = {}
            for p in starters:
                pos = p.get("position")
                positions[pos] = positions.get(pos, 0) + 1
            print(positions)
            
            # Check specific rule: 1 GK
            if positions.get("GKP", 0) == 1:
                print("SUCCESS: 1 GK in starting 11.")
            else:
                print(f"FAILURE: {positions.get('GKP', 0)} GKs in starting 11.")
                
        else:
            print("FAILURE: Incorrect counts.")
            
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"Exception: {e}")
