import requests
import time

# FPL rejects clients that do not send a browser-like user agent.
FPL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FPLLineupOptimizer/1.0)"
}
MAX_BANKED_TRANSFERS = 5


def banked_transfers_for_next_deadline(events, chips=None):
    """
    Free transfers available at the next deadline.

    Each completed gameweek grants one transfer, capped at 5. Transfers spend
    that bank. Wildcard and Free Hit do not spend it.
    """
    chips_by_gw = {}
    for chip in chips or []:
        chips_by_gw[chip.get("event")] = (chip.get("name") or "").lower()

    available = 1
    history = list(events or [])
    if not history:
        return available

    for index, event in enumerate(history):
        made = int(event.get("event_transfers") or 0)
        chip = chips_by_gw.get(event.get("event"), "")
        if chip in ("wildcard", "freehit"):
            made = 0
        remaining = max(0, available - made)
        available = min(MAX_BANKED_TRANSFERS, remaining + 1)
        if index == len(history) - 1:
            return available
    return available


class FPLService:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
        self.CACHE_DURATION = 3600  # 1 hour

    def _get_json(self, url):
        response = requests.get(url, headers=FPL_HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_latest_data(self):
        """Fetches bootstrap-static and fixtures data."""
        if self._is_cache_valid("bootstrap-static") and self._is_cache_valid("fixtures"):
            return {
                "static": self._cache["bootstrap-static"],
                "fixtures": self._cache["fixtures"]
            }

        # Fetch Bootstrap Static
        try:
            static_data = self._get_json(f"{self.BASE_URL}/bootstrap-static/")
            self._update_cache("bootstrap-static", static_data)

            # Fetch Fixtures
            fixtures_data = self._get_json(f"{self.BASE_URL}/fixtures/")
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

        urls = [f"{self.BASE_URL}/entry/{manager_id}/event/{gw}/picks/"]
        if gw > 1:
            urls.append(f"{self.BASE_URL}/entry/{manager_id}/event/{gw - 1}/picks/")
        last_error = None
        for url in urls:
            try:
                return self._get_json(url)
            except requests.RequestException as e:
                last_error = e
                print(f"Error fetching manager team: {e}")
        raise last_error

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
            data = self._get_json(f"{self.BASE_URL}/entry/{manager_id}/history/")
            
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
                'available': available_chips,
                'banked_transfers': banked_transfers_for_next_deadline(
                    data.get('current') or [],
                    data.get('chips') or [],
                ),
            }
        except requests.RequestException as e:
            print(f"Error fetching manager chips: {e}")
            # Return all chips as available if API fails
            return {
                'used': [],
                'available': ['wildcard', 'free_hit', 'bench_boost', 'triple_captain'],
                'banked_transfers': 1,
            }
