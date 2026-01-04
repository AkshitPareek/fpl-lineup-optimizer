import requests
import json
import time

class FPLService:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
        self.CACHE_DURATION = 3600  # 1 hour

    def get_latest_data(self):
        """Fetches bootstrap-static and fixtures data."""
        if self._is_cache_valid("bootstrap-static") and self._is_cache_valid("fixtures"):
            return {
                "static": self._cache["bootstrap-static"],
                "fixtures": self._cache["fixtures"]
            }

        # Fetch Bootstrap Static
        try:
            static_response = requests.get(f"{self.BASE_URL}/bootstrap-static/")
            static_response.raise_for_status()
            static_data = static_response.json()
            self._update_cache("bootstrap-static", static_data)

            # Fetch Fixtures
            fixtures_response = requests.get(f"{self.BASE_URL}/fixtures/")
            fixtures_response.raise_for_status()
            fixtures_data = fixtures_response.json()
            self._update_cache("fixtures", fixtures_data)
            
            return {
                "static": static_data,
                "fixtures": fixtures_data
            }
        except requests.RequestException as e:
            print(f"Error fetching FPL data: {e}")
            raise

    def get_manager_team(self, manager_id: int):
        """Fetches a manager's current team."""
        # Get current gameweek
        static_data = self.get_latest_data()["static"]
        current_event = next((e for e in static_data["events"] if e["is_current"]), None)
        
        if not current_event:
            # If no current event (e.g. pre-season), try first event or handle error
            # For now, let's assume season is active or use the next event ID - 1
            next_event = next((e for e in static_data["events"] if e["is_next"]), None)
            gw = next_event["id"] - 1 if next_event else 38
        else:
            gw = current_event["id"]

        try:
            url = f"{self.BASE_URL}/entry/{manager_id}/event/{gw}/picks/"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
             # Fallback for previous GW if current hasn't started/picks not public?
             # But usually picks are public after deadline.
             print(f"Error fetching manager team: {e}")
             raise

    def _update_cache(self, key, data):
        self._cache[key] = data
        self._cache_expiry[key] = time.time() + self.CACHE_DURATION

    def _is_cache_valid(self, key):
        return key in self._cache and time.time() < self._cache_expiry.get(key, 0)
    
    def get_manager_chips(self, manager_id: int):
        """
        Fetches a manager's chip usage history from FPL API.
        Returns dict with 'used' and 'available' chips.
        """
        all_chips = ['wildcard', 'freehit', 'bboost', 'triple_captain']
        
        try:
            url = f"{self.BASE_URL}/entry/{manager_id}/history/"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # 'chips' field contains list of used chips with 'name' and 'event' (GW)
            used_chips = []
            for chip in data.get('chips', []):
                chip_name = chip.get('name', '').lower()
                used_chips.append({
                    'name': chip_name,
                    'gameweek': chip.get('event')
                })
            
            # Map FPL chip names to our names
            chip_name_map = {
                'wildcard': 'wildcard',
                '3xc': 'triple_captain',
                'bboost': 'bench_boost',
                'freehit': 'free_hit'
            }
            
            used_names = [c['name'] for c in used_chips]
            available_chips = [
                chip_name_map.get(c, c) 
                for c in all_chips 
                if c not in used_names
            ]
            
            return {
                'used': used_chips,
                'available': available_chips
            }
        except requests.RequestException as e:
            print(f"Error fetching manager chips: {e}")
            # Return all chips as available if API fails
            return {
                'used': [],
                'available': ['wildcard', 'free_hit', 'bench_boost', 'triple_captain']
            }
