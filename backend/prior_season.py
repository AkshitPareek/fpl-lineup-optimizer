"""
Last-season totals for every current player.

FPL only exposes previous seasons through each player's element-summary, so
the totals are fetched once per season in the background and cached on disk.
Rows are keyed by the player `code`, which stays the same across seasons
(element ids do not).
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
PRIOR_FIELDS = ("prior_points", "prior_minutes", "prior_starts")


def previous_season_name(events: List[Dict]) -> Optional[str]:
    """'2025/26' while the 2026/27 season is being played."""
    deadlines = [e.get("deadline_time") for e in events or [] if e.get("deadline_time")]
    if not deadlines:
        return None
    start_year = int(min(deadlines)[:4]) - 1
    return f"{start_year}/{str(start_year + 1)[2:]}"


def totals_from_summary(summary: Dict, season_name: str) -> Optional[Dict[str, int]]:
    for past in summary.get("history_past") or []:
        if past.get("season_name") == season_name:
            return {
                "prior_points": int(past.get("total_points") or 0),
                "prior_minutes": int(past.get("minutes") or 0),
                "prior_starts": int(past.get("starts") or 0),
            }
    return None


class PriorSeasonStats:
    def __init__(self, cache_dir: str = CACHE_DIR, workers: int = 1):
        self.cache_dir = cache_dir
        self.workers = workers
        self._by_code: Dict[str, Dict[str, int]] = {}
        self._season: Optional[str] = None
        self._lock = threading.Lock()
        self._loading = False
        self._loader: Optional[threading.Thread] = None

    def _cache_path(self, season_name: str) -> str:
        return os.path.join(self.cache_dir, f"prior_season_{season_name.replace('/', '-')}.json")

    def _fetch_one(self, element: Dict, season_name: str):
        response = requests.get(f"{BASE_URL}/element-summary/{element['id']}/", headers=HEADERS, timeout=20)
        response.raise_for_status()
        return str(element["code"]), totals_from_summary(response.json(), season_name)

    def _load(self, elements: List[Dict], season_name: str):
        path = self._cache_path(season_name)
        try:
            if os.path.exists(path):
                with open(path) as f:
                    by_code = json.load(f)
            else:
                by_code = {}
                failures = 0
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    futures = [pool.submit(self._fetch_one, e, season_name) for e in elements]
                    for future in futures:
                        try:
                            code, totals = future.result()
                        except requests.RequestException as e:
                            failures += 1
                            print(f"Prior-season fetch failed: {e}")
                            continue
                        if totals:
                            by_code[code] = totals
                if by_code:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(by_code, f)
            with self._lock:
                self._by_code = by_code
                self._season = season_name
            print(f"Loaded {season_name} totals for {len(by_code)} players")
        finally:
            with self._lock:
                self._loading = False

    def attach(self, static_data: Dict, wait: float = 0) -> None:
        """
        Add prior_points / prior_minutes / prior_starts to each element in place.

        The first call starts the background load and waits up to `wait`
        seconds for it; until it finishes, predictions shrink toward the
        newcomer baseline instead of last season.
        """
        if os.getenv("LOAD_PRIOR_SEASON", "1") == "0":
            return
        season_name = previous_season_name(static_data.get("events", []))
        if not season_name:
            return
        cached = self._read_cache(season_name)
        if cached is not None:
            with self._lock:
                self._by_code = cached
                self._season = season_name
            self._apply(static_data, cached)
            return
        with self._lock:
            if self._season != season_name and not self._loading:
                self._loading = True
                self._loader = threading.Thread(
                    target=self._load,
                    args=(list(static_data.get("elements", [])), season_name),
                    daemon=True,
                )
                self._loader.start()
            loader = self._loader
        if wait and loader:
            loader.join(timeout=wait)
        with self._lock:
            if self._season != season_name:
                return
            by_code = self._by_code
        self._apply(static_data, by_code)

    def _bundled_path(self, season_name: str) -> str:
        name = f"prior_season_{season_name.replace('/', '-')}.json"
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)

    def _read_cache(self, season_name: str) -> Optional[Dict[str, Dict[str, int]]]:
        for path in (self._cache_path(season_name), self._bundled_path(season_name)):
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        return None

    def _apply(self, static_data: Dict, by_code: Dict[str, Dict[str, int]]) -> None:
        for element in static_data.get("elements", []):
            totals = by_code.get(str(element.get("code")))
            for field in PRIOR_FIELDS:
                element[field] = totals[field] if totals else None
