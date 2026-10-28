import requests
import time

from prior_season import PriorSeasonStats

# FPL rejects clients that do not send a browser-like user agent.
FPL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FPLLineupOptimizer/1.0)"
}
MAX_BANKED_TRANSFERS = 5
LAST_FIRST_HALF_GAMEWEEK = 19
# The first request after a cold start waits this long for last season's totals.
PRIOR_SEASON_WAIT_SECONDS = 30


UNLIMITED_TRANSFER_CHIPS = ("wildcard", "freehit")


def _chips_by_gameweek(chips):
    return {chip.get("event"): (chip.get("name") or "").lower() for chip in chips or []}


def banked_transfers_for_next_deadline(events, chips=None):
    """
    Free transfers available at the next deadline.

    Transfers before a manager's first deadline are unlimited, then each
    gameweek grants one more, capped at 5. A Wildcard or Free Hit week keeps
    the saved transfers but grants none: 2 saved before the chip are still 2
    the week after.
    """
    chips_by_gw = _chips_by_gameweek(chips)
    available = 1
    for event in list(events or [])[1:]:
        if chips_by_gw.get(event.get("event")) in UNLIMITED_TRANSFER_CHIPS:
            continue
        made = int(event.get("event_transfers") or 0)
        available = min(MAX_BANKED_TRANSFERS, max(0, available - made) + 1)
    return available


def selling_price(purchase_price, current_price):
    """FPL keeps half of any rise, rounded down to £0.1m; a fall is passed on in full. Prices in tenths."""
    if current_price <= purchase_price:
        return current_price
    return purchase_price + (current_price - purchase_price) // 2


def manager_squad_state(history, transfers, picks_by_gw, elements, initial_price=None):
    """
    The squad, bank, selling prices and free transfers a manager takes into the next deadline.

    history: /entry/{id}/history/; transfers: /entry/{id}/transfers/;
    picks_by_gw: callable gw -> /entry/{id}/event/{gw}/picks/; elements: bootstrap elements.
    initial_price: callable player id -> price in tenths when the season-start squad was
    picked after GW1; defaults to the season start price.
    Transfers already made for the next deadline are applied.
    """
    events = list(history.get("current") or [])
    if not events:
        return None
    chips_by_gw = _chips_by_gameweek(history.get("chips"))
    by_id = {e["id"]: e for e in elements}

    # A Free Hit squad reverts at the next deadline, so plan from the week before it.
    base = events[-1]
    if chips_by_gw.get(base["event"]) == "freehit" and len(events) > 1:
        base = events[-2]
    squad = [p["element"] for p in picks_by_gw(base["event"]).get("picks", [])]
    bank = int(base.get("bank") or 0)
    free_transfers = banked_transfers_for_next_deadline(events, history.get("chips"))

    last_event = events[-1]["event"]
    pending = sorted((t for t in transfers or [] if (t.get("event") or 0) > last_event),
                     key=lambda t: t.get("time") or "")
    pending_unlimited = any(chips_by_gw.get(t.get("event")) in UNLIMITED_TRANSFER_CHIPS for t in pending)
    for t in pending:
        if t["element_out"] in squad:
            squad[squad.index(t["element_out"])] = t["element_in"]
            bank += int(t["element_out_cost"]) - int(t["element_in_cost"])
    if not pending_unlimited:
        free_transfers = max(0, free_transfers - len(pending))

    purchase = {}
    for t in sorted(transfers or [], key=lambda t: t.get("time") or ""):
        if chips_by_gw.get(t.get("event")) == "freehit":
            continue
        purchase[t["element_in"]] = int(t["element_in_cost"])

    selling = {}
    for pid in squad:
        element = by_id.get(pid)
        if element is None:
            continue
        now = int(element["now_cost"])
        bought = purchase.get(pid)
        if bought is None:
            bought = initial_price(pid) if initial_price else now - int(element.get("cost_change_start") or 0)
        selling[pid] = selling_price(bought, now) / 10.0

    return {
        "squad": squad,
        "selling_prices": selling,
        "bank": bank / 10.0,
        "free_transfers": free_transfers,
        "pending_transfers": len(pending),
    }


class FPLService:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
        self.CACHE_DURATION = 3600  # 1 hour
        self.prior_season = PriorSeasonStats()

    def _get_json(self, url):
        response = requests.get(url, headers=FPL_HEADERS, timeout=12)
        response.raise_for_status()
        return response.json()

    def get_latest_data(self, prior_wait: float = 0):
        """Bootstrap and fixtures. Last season loads in the background and does not hold the request."""
        if self._is_cache_valid("bootstrap-static") and self._is_cache_valid("fixtures"):
            self.prior_season.attach(self._cache["bootstrap-static"], wait=prior_wait)
            return {
                "static": self._cache["bootstrap-static"],
                "fixtures": self._cache["fixtures"]
            }

        try:
            static_data = self._get_json(f"{self.BASE_URL}/bootstrap-static/")
            self._update_cache("bootstrap-static", static_data)
            fixtures_data = self._get_json(f"{self.BASE_URL}/fixtures/")
            self._update_cache("fixtures", fixtures_data)
            self.prior_season.attach(static_data, wait=prior_wait)
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

    def get_manager_squad_state(self, manager_id: int, history=None):
        """Squad, bank, selling prices and free transfers for the next deadline (see manager_squad_state)."""
        base = f"{self.BASE_URL}/entry/{manager_id}"
        history = history or self._get_json(f"{base}/history/")
        transfers = self._get_json(f"{base}/transfers/")
        elements = self.get_latest_data()["static"]["elements"]
        events = history.get("current") or []
        initial_price = None
        if events and events[0]["event"] > 1:
            first_gw = events[0]["event"]

            def initial_price(pid):
                summary = self._get_json(f"{self.BASE_URL}/element-summary/{pid}/")
                for match in summary.get("history", []):
                    if match.get("round") == first_gw:
                        return int(match["value"])
                return next(int(e["now_cost"]) for e in elements if e["id"] == pid)

        return manager_squad_state(
            history,
            transfers,
            lambda gw: self._get_json(f"{base}/event/{gw}/picks/"),
            elements,
            initial_price,
        )

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
        all_chips = ['wildcard', 'freehit', 'bboost', '3xc']
        
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
            
            history = data.get('current') or []
            # Every chip comes twice: one set up to the GW19 deadline, a fresh set from GW20.
            next_gw = history[-1]['event'] + 1 if history else 1
            in_this_half = (lambda gw: gw <= LAST_FIRST_HALF_GAMEWEEK) if next_gw <= LAST_FIRST_HALF_GAMEWEEK \
                else (lambda gw: gw > LAST_FIRST_HALF_GAMEWEEK)
            used_names = [c['name'] for c in used_chips if in_this_half(c['gameweek'] or 0)]
            available_chips = [
                chip_name_map.get(c, c) 
                for c in all_chips 
                if c not in used_names
            ]
            
            result = {
                'used': used_chips,
                'used_this_half': [chip_name_map.get(c, c) for c in used_names],
                'available': available_chips,
                'banked_transfers': banked_transfers_for_next_deadline(
                    history,
                    data.get('chips') or [],
                ),
                'bank': history[-1].get('bank', 0) / 10.0 if history else None,
            }
            if history:
                try:
                    state = self.get_manager_squad_state(manager_id, history=data)
                    result['banked_transfers'] = state['free_transfers']
                    result['bank'] = state['bank']
                    result['pending_transfers'] = state['pending_transfers']
                except (requests.RequestException, KeyError, StopIteration) as e:
                    print(f"Error reading manager transfers: {e}")
            return result
        except requests.RequestException as e:
            print(f"Error fetching manager chips: {e}")
            # Return all chips as available if API fails
            return {
                'used': [],
                'used_this_half': [],
                'available': ['wildcard', 'free_hit', 'bench_boost', 'triple_captain'],
                'banked_transfers': 1,
            }
