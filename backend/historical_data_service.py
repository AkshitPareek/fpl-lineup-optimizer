"""
Historical Data Service for FPL Backtesting

Responsible for fetching and reconstructing the state of FPL data at past points in time.
Uses 'element-summary' endpoint to get per-gameweek history for all players.
Implements caching to avoid excessive API calls.
"""

import requests
import json
import os
import pandas as pd
import time
from typing import Dict, List, Any, Optional

class HistoricalDataService:
    BASE_URL = "https://fantasy.premierleague.com/api"
    CACHE_DIR = "data/cache"
    HISTORY_FILE = "data/cache/all_player_history.json"
    CACHE_DURATION = 86400 * 7  # 1 week cache for history (doesn't change often for past)
    
    def __init__(self):
        # Ensure cache directory exists
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._history_cache = self._load_history_cache()
    
    def _load_history_cache(self) -> Dict[str, Any]:
        """Load cached history data if available."""
        if os.path.exists(self.HISTORY_FILE):
            try:
                # Check modification time? For now, just load.
                with open(self.HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history cache: {e}")
        return {}
    
    def _save_history_cache(self):
        """Save history cache to disk."""
        try:
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(self._history_cache, f)
        except Exception as e:
            print(f"Error saving history cache: {e}")

    def fetch_all_player_history(self, elements: List[Dict]):
        """
        Fetch full history for all players in the list.
        
        Args:
            elements: List of player dicts (from bootstrap-static)
        """
        updates_made = False
        total_players = len(elements)
        print(f"Fetching history for {total_players} players...")
        
        for idx, player in enumerate(elements):
            pid = str(player['id'])
            
            # Skip if already cached and valid?
            # For backtesting, we might need fresh data if new GWs happened.
            # But historical GWs don't change. 
            # Only fetch if not present or explicitly requested?
            # For this implementation, we fetch if missing.
            
            if pid not in self._history_cache:
                try:
                    time.sleep(0.05)  # Rate limiting
                    response = requests.get(f"{self.BASE_URL}/element-summary/{pid}/")
                    response.raise_for_status()
                    data = response.json()
                    self._history_cache[pid] = data
                    updates_made = True
                    
                    if idx % 50 == 0:
                        print(f"Fetched {idx}/{total_players} players")
                        self._save_history_cache()  # Save incrementally
                except Exception as e:
                    print(f"Failed to fetch history for player {pid}: {e}")
        
        if updates_made:
            self._save_history_cache()
            print("History cache updated.")

    def get_gameweek_state(self, 
                           gameweek: int, 
                           static_data: Dict) -> pd.DataFrame:
        """
        Reconstruct player state as it was before 'gameweek' deadline.
        
        Args:
            gameweek: The target gameweek (1-38)
            static_data: Current bootstrap-static data
            
        Returns:
            DataFrame of players with reconstructed attributes:
            - now_cost (at that time)
            - form (calculated from previous matches)
            - points_per_game (calculated)
            - total_points (up to that time)
        """
        elements = static_data['elements']
        
        # Ensure we have history
        if not self._history_cache:
            self.fetch_all_player_history(elements)
            
        reconstructed = []
        
        for player in elements:
            pid = str(player['id'])
            history = self._history_cache.get(pid, {}).get('history', [])
            
            # Filter history for rounds BEFORE the target gameweek
            # We want state entering the GW, so include rounds < gameweek
            past_rounds = [r for r in history if r['round'] < gameweek]
            
            # Calculate stats based on past rounds
            total_points = sum(r['total_points'] for r in past_rounds)
            minutes = sum(r['minutes'] for r in past_rounds)
            games_played = len([r for r in past_rounds if r['minutes'] > 0])
            
            # Accumulate advanced metrics (full season)
            influence = sum(float(r.get('influence', 0)) for r in past_rounds)
            creativity = sum(float(r.get('creativity', 0)) for r in past_rounds)
            threat = sum(float(r.get('threat', 0)) for r in past_rounds)
            ict_index = sum(float(r.get('ict_index', 0)) for r in past_rounds)
            expected_goals = sum(float(r.get('expected_goals', 0)) for r in past_rounds)
            expected_assists = sum(float(r.get('expected_assists', 0)) for r in past_rounds)
            
            # RECENT metrics (last 6 games) - more relevant for current form
            recent_rounds = [r for r in past_rounds if r['minutes'] > 0][-6:]
            recent_xg = sum(float(r.get('expected_goals', 0)) for r in recent_rounds)
            recent_xa = sum(float(r.get('expected_assists', 0)) for r in recent_rounds)
            recent_minutes = sum(r['minutes'] for r in recent_rounds)
            
            # Recent minutes (last 4 games) - for rotation risk detection
            last_4_rounds = [r for r in past_rounds][-4:]
            recent_4_minutes = sum(r['minutes'] for r in last_4_rounds)
            
            points_per_game = (total_points / games_played) if games_played > 0 else 0
            
            # Form: Average points in last 30 days? 
            # FPL defines form as average points per match over last 30 days.
            # Simplified: Average over last 4 matches.
            last_4_matches = [r for r in past_rounds if r['minutes'] > 0][-4:]
            if last_4_matches:
                form = sum(r['total_points'] for r in last_4_matches) / len(last_4_matches)
            else:
                form = 0.0
                
            # Price: Get value from the last completed round, or initial price
            if past_rounds:
                # 'value' in history is price * 10
                now_cost = past_rounds[-1]['value']
            else:
                # If no games played, calculate from current cost - cost_change_start?
                # Approximation: use current cost (not perfect but acceptable fallback)
                now_cost = player['now_cost'] 
            
            # Create reconstructed player object
            recon_player = player.copy()
            recon_player.update({
                'now_cost': now_cost,
                'form': f"{form:.1f}",
                'points_per_game': f"{points_per_game:.1f}",
                'total_points': total_points,
                'minutes': minutes,
                'influence': f"{influence:.1f}",
                'creativity': f"{creativity:.1f}",
                'threat': f"{threat:.1f}",
                'ict_index': f"{ict_index:.1f}",
                'expected_goals': f"{expected_goals:.2f}",
                'expected_assists': f"{expected_assists:.2f}",
                # NEW: Recent metrics for form-weighted predictions
                'recent_xg': f"{recent_xg:.2f}",
                'recent_xa': f"{recent_xa:.2f}",
                'recent_minutes': recent_minutes,
                'recent_4_minutes': recent_4_minutes,
                # Clear future-looking ep_next
                'ep_next': f"{form:.1f}", 
                'ep_this': None,
                'event_points': 0 # Reset for the target GW
            })
            
            reconstructed.append(recon_player)
            
        return pd.DataFrame(reconstructed)

    def get_actual_score(self, 
                         player_id: int, 
                         gameweek: int) -> float:
        """Get actual points scored by a player in a specific gameweek."""
        pid = str(player_id)
        history = self._history_cache.get(pid, {}).get('history', [])
        
        match = next((r for r in history if r['round'] == gameweek), None)
        if match:
            return float(match['total_points'])
        return 0.0
