import requests
import json

response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
data = response.json()
player = data['elements'][0]
print(json.dumps(list(player.keys()), indent=2))
print(f"Sample ep_next: {player.get('ep_next')}")
print(f"Sample ep_this: {player.get('ep_this')}")
