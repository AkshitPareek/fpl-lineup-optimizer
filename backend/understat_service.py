"""
Understat Service for FPL Optimizer

Fetches xG, xA, and xGI data from Understat to enhance point predictions.
Uses the understatapi library to scrape data from understat.com.

Data available:
- Per-match xG/xA/xGI for each player
- Season aggregates
- Team-level expected metrics
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os
import time
import logging

try:
    from understatapi import UnderstatClient
    UNDERSTAT_AVAILABLE = True
except ImportError:
    UNDERSTAT_AVAILABLE = False
    logging.warning("understatapi not installed. Run: pip install understatapi")

# Fuzzy matching for player names
try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("rapidfuzz not installed. Run: pip install rapidfuzz")


@dataclass
class PlayerXGStats:
    """xG statistics for a player."""
    player_id: int  # Understat ID
    player_name: str
    team: str
    games: int
    goals: int
    xG: float
    assists: int
    xA: float
    shots: int
    key_passes: int
    xGI: float  # xG + xA
    npg: int  # Non-penalty goals
    npxG: float  # Non-penalty xG
    xG_per_90: float
    xA_per_90: float
    minutes: int


class UnderstatService:
    """
    Service for fetching xG/xA data from Understat.
    
    Features:
    - Fetch player stats for EPL season
    - Match players between Understat and FPL IDs
    - Provide enriched data for point predictions
    """
    
    CACHE_DIR = "data/cache/understat"
    CACHE_DURATION = 86400  # 24 hours
    
    # Team name mappings between FPL and Understat
    TEAM_MAPPINGS = {
        # FPL short name -> Understat name
        "ARS": "Arsenal",
        "AVL": "Aston Villa",
        "BOU": "Bournemouth",
        "BRE": "Brentford",
        "BHA": "Brighton",
        "CHE": "Chelsea",
        "CRY": "Crystal Palace",
        "EVE": "Everton",
        "FUL": "Fulham",
        "IPS": "Ipswich",
        "LEI": "Leicester",
        "LIV": "Liverpool",
        "MCI": "Manchester City",
        "MUN": "Manchester United",
        "NEW": "Newcastle United",
        "NFO": "Nottingham Forest",
        "SOU": "Southampton",
        "TOT": "Tottenham",
        "WHU": "West Ham",
        "WOL": "Wolverhampton Wanderers",
    }
    
    # Reverse mapping
    UNDERSTAT_TO_FPL = {v: k for k, v in TEAM_MAPPINGS.items()}
    
    def __init__(self):
        """Initialize the Understat service."""
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._player_id_mapping: Dict[int, int] = {}  # FPL ID -> Understat ID
        
        # Create cache directory
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Load cached mappings if available
        self._load_id_mappings()
    
    def _load_id_mappings(self):
        """Load player ID mappings from cache."""
        mapping_file = os.path.join(self.CACHE_DIR, "id_mappings.json")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    self._player_id_mapping = {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                logging.warning(f"Failed to load ID mappings: {e}")
    
    def _save_id_mappings(self):
        """Save player ID mappings to cache."""
        mapping_file = os.path.join(self.CACHE_DIR, "id_mappings.json")
        try:
            with open(mapping_file, 'w') as f:
                json.dump(self._player_id_mapping, f)
        except Exception as e:
            logging.warning(f"Failed to save ID mappings: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        return key in self._cache and time.time() < self._cache_expiry.get(key, 0)
    
    def _update_cache(self, key: str, data: Any):
        """Update cache with new data."""
        self._cache[key] = data
        self._cache_expiry[key] = time.time() + self.CACHE_DURATION
    
    def get_league_players(self, season: str = "2025") -> List[Dict]:
        """
        Fetch all player stats for EPL season.
        
        Args:
            season: Season year (e.g., "2025" for 2025/26)
            
        Returns:
            List of player stat dictionaries
        """
        if not UNDERSTAT_AVAILABLE:
            logging.error("understatapi not available")
            return []
        
        cache_key = f"epl_players_{season}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with UnderstatClient() as client:
                players = client.league(league="EPL").get_player_data(season=season)
            
            self._update_cache(cache_key, players)
            
            # Also cache to disk
            cache_file = os.path.join(self.CACHE_DIR, f"epl_players_{season}.json")
            with open(cache_file, 'w') as f:
                json.dump(players, f)
            
            return players
            
        except Exception as e:
            logging.error(f"Failed to fetch Understat data: {e}")
            
            # Try loading from disk cache
            cache_file = os.path.join(self.CACHE_DIR, f"epl_players_{season}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            return []
    
    def get_player_matches(self, understat_id: int) -> List[Dict]:
        """
        Fetch match-by-match stats for a player.
        
        Args:
            understat_id: Player's Understat ID
            
        Returns:
            List of match stat dictionaries
        """
        if not UNDERSTAT_AVAILABLE:
            return []
        
        cache_key = f"player_matches_{understat_id}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with UnderstatClient() as client:
                matches = client.player(player=understat_id).get_match_data()
            
            self._update_cache(cache_key, matches)
            return matches
            
        except Exception as e:
            logging.error(f"Failed to fetch player matches: {e}")
            return []
    
    def match_player_name(self, fpl_name: str, fpl_team: str, 
                          understat_players: List[Dict]) -> Optional[Dict]:
        """
        Match an FPL player to their Understat record using fuzzy matching.
        
        Args:
            fpl_name: Player's FPL web_name
            fpl_team: Player's FPL team short name
            understat_players: List of Understat player records
            
        Returns:
            Matched Understat player record or None
        """
        if not FUZZY_AVAILABLE:
            # Fallback to substring matching
            for player in understat_players:
                us_name = player.get('player_name', '').lower()
                fpl_lower = fpl_name.lower()
                # Check both directions: FPL name in Understat name or vice versa
                if fpl_lower in us_name or us_name.split()[-1] in fpl_lower:
                    return player
            return None
        
        # Get Understat team name
        understat_team = self.TEAM_MAPPINGS.get(fpl_team)
        
        # First try: filter by team
        if understat_team:
            team_players = [
                p for p in understat_players 
                if p.get('team_title', '') == understat_team
            ]
            
            if team_players:
                # Try to find match within team
                player_names = [p.get('player_name', '') for p in team_players]
                match = process.extractOne(
                    fpl_name, 
                    player_names, 
                    scorer=fuzz.token_sort_ratio
                )
                
                if match and match[1] >= 60:  # Lower threshold for team-filtered search
                    matched_name = match[0]
                    for player in team_players:
                        if player.get('player_name') == matched_name:
                            return player
        
        # Second try: search all players (for cases like "Haaland" vs "Erling Haaland")
        all_names = [p.get('player_name', '') for p in understat_players]
        
        # Try partial ratio for better matching of short names
        match = process.extractOne(
            fpl_name, 
            all_names, 
            scorer=fuzz.partial_ratio
        )
        
        if match and match[1] >= 80:  # Higher threshold for global search
            matched_name = match[0]
            for player in understat_players:
                if player.get('player_name') == matched_name:
                    return player
        
        # Third try: token set ratio (handles name order differences)
        match = process.extractOne(
            fpl_name, 
            all_names, 
            scorer=fuzz.token_set_ratio
        )
        
        if match and match[1] >= 75:
            matched_name = match[0]
            for player in understat_players:
                if player.get('player_name') == matched_name:
                    return player
        
        return None
    
    def sync_with_fpl(self, fpl_players_df: pd.DataFrame, 
                      teams_df: pd.DataFrame,
                      season: str = "2025") -> pd.DataFrame:
        """
        Enrich FPL player data with Understat xG/xA metrics.
        
        Args:
            fpl_players_df: FPL players DataFrame
            teams_df: FPL teams DataFrame
            season: Understat season
            
        Returns:
            Enriched DataFrame with Understat columns
        """
        # Fetch Understat data
        understat_players = self.get_league_players(season)
        
        if not understat_players:
            logging.warning("No Understat data available, returning original DataFrame")
            return fpl_players_df
        
        # Build team name lookup
        team_names = {row['id']: row['short_name'] for _, row in teams_df.iterrows()}
        
        # Prepare enrichment columns
        enriched_data = []
        
        for _, player in fpl_players_df.iterrows():
            fpl_id = player['id']
            fpl_name = player.get('web_name', '')
            fpl_team = team_names.get(player.get('team'), '')
            
            # Check cached mapping first
            if fpl_id in self._player_id_mapping:
                understat_id = self._player_id_mapping[fpl_id]
                understat_player = next(
                    (p for p in understat_players if int(p.get('id', 0)) == understat_id),
                    None
                )
            else:
                # Try to match
                understat_player = self.match_player_name(fpl_name, fpl_team, understat_players)
                
                # Cache the mapping
                if understat_player:
                    self._player_id_mapping[fpl_id] = int(understat_player.get('id', 0))
            
            if understat_player:
                xG = float(understat_player.get('xG', 0))
                xA = float(understat_player.get('xA', 0))
                games = int(understat_player.get('games', 0))
                mins = int(understat_player.get('time', 0))
                
                enriched_data.append({
                    'id': fpl_id,
                    'understat_id': int(understat_player.get('id', 0)),
                    'understat_xG': xG,
                    'understat_xA': xA,
                    'understat_xGI': xG + xA,
                    'understat_npxG': float(understat_player.get('npxG', 0)),
                    'understat_shots': int(understat_player.get('shots', 0)),
                    'understat_key_passes': int(understat_player.get('key_passes', 0)),
                    'understat_games': games,
                    'understat_minutes': mins,
                    'understat_xG_per_90': (xG / max(1, mins)) * 90 if mins > 0 else 0,
                    'understat_xA_per_90': (xA / max(1, mins)) * 90 if mins > 0 else 0,
                    'understat_matched': True
                })
            else:
                enriched_data.append({
                    'id': fpl_id,
                    'understat_id': None,
                    'understat_xG': 0.0,
                    'understat_xA': 0.0,
                    'understat_xGI': 0.0,
                    'understat_npxG': 0.0,
                    'understat_shots': 0,
                    'understat_key_passes': 0,
                    'understat_games': 0,
                    'understat_minutes': 0,
                    'understat_xG_per_90': 0.0,
                    'understat_xA_per_90': 0.0,
                    'understat_matched': False
                })
        
        # Save mappings
        self._save_id_mappings()
        
        # Merge with original DataFrame
        enriched_df = pd.DataFrame(enriched_data)
        result = fpl_players_df.merge(enriched_df, on='id', how='left')
        
        return result
    
    def get_team_stats(self, team_name: str, season: str = "2025") -> Optional[Dict]:
        """
        Fetch team-level xG statistics.
        
        Args:
            team_name: FPL team short name (e.g., "ARS")
            season: Season year
            
        Returns:
            Team stats dictionary or None
        """
        if not UNDERSTAT_AVAILABLE:
            return None
        
        understat_team = self.TEAM_MAPPINGS.get(team_name)
        if not understat_team:
            return None
        
        cache_key = f"team_{understat_team}_{season}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with UnderstatClient() as client:
                # Get team data
                team_data = client.team(team=understat_team).get_team_data(season=season)
            
            self._update_cache(cache_key, team_data)
            return team_data
            
        except Exception as e:
            logging.error(f"Failed to fetch team stats: {e}")
            return None
    
    def calculate_regression_factor(self, player_stats: Dict) -> float:
        """
        Calculate xG regression factor to identify over/under-performers.
        
        A player scoring more than their xG indicates overperformance,
        while scoring less indicates underperformance (potential for regression).
        
        Args:
            player_stats: Understat player statistics
            
        Returns:
            Regression factor (>1 = underperforming, <1 = overperforming)
        """
        goals = float(player_stats.get('goals', 0))
        xG = float(player_stats.get('xG', 0))
        
        if xG < 0.5:  # Not enough data
            return 1.0
        
        # Ratio of xG to actual goals
        # >1 means underperforming (should regress up)
        # <1 means overperforming (should regress down)
        regression_factor = xG / max(0.1, goals)
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, regression_factor))
    
    def get_recent_form_xG(self, understat_id: int, last_n_matches: int = 5) -> Dict:
        """
        Get xG/xA form over recent matches.
        
        Args:
            understat_id: Player's Understat ID
            last_n_matches: Number of recent matches to consider
            
        Returns:
            Dict with recent xG/xA averages
        """
        matches = self.get_player_matches(understat_id)
        
        if not matches:
            return {
                'recent_xG': 0.0,
                'recent_xA': 0.0,
                'recent_xG_per_90': 0.0,
                'recent_xA_per_90': 0.0,
                'matches_analyzed': 0
            }
        
        # Sort by date (most recent first)
        sorted_matches = sorted(
            matches, 
            key=lambda x: x.get('date', ''), 
            reverse=True
        )[:last_n_matches]
        
        total_xG = sum(float(m.get('xG', 0)) for m in sorted_matches)
        total_xA = sum(float(m.get('xA', 0)) for m in sorted_matches)
        total_mins = sum(int(m.get('time', 0)) for m in sorted_matches)
        
        return {
            'recent_xG': total_xG,
            'recent_xA': total_xA,
            'recent_xG_per_90': (total_xG / max(1, total_mins)) * 90 if total_mins > 0 else 0,
            'recent_xA_per_90': (total_xA / max(1, total_mins)) * 90 if total_mins > 0 else 0,
            'matches_analyzed': len(sorted_matches)
        }
    
    def get_all_stats_summary(self, season: str = "2025") -> pd.DataFrame:
        """
        Get a summary DataFrame of all player xG stats.
        
        Returns:
            DataFrame with player xG/xA statistics
        """
        players = self.get_league_players(season)
        
        if not players:
            return pd.DataFrame()
        
        records = []
        for p in players:
            mins = int(p.get('time', 0))
            xG = float(p.get('xG', 0))
            xA = float(p.get('xA', 0))
            
            records.append({
                'understat_id': int(p.get('id', 0)),
                'player_name': p.get('player_name', ''),
                'team': p.get('team_title', ''),
                'games': int(p.get('games', 0)),
                'minutes': mins,
                'goals': int(p.get('goals', 0)),
                'xG': xG,
                'npxG': float(p.get('npxG', 0)),
                'assists': int(p.get('assists', 0)),
                'xA': xA,
                'xGI': xG + xA,
                'shots': int(p.get('shots', 0)),
                'key_passes': int(p.get('key_passes', 0)),
                'xG_per_90': (xG / max(1, mins)) * 90 if mins > 0 else 0,
                'xA_per_90': (xA / max(1, mins)) * 90 if mins > 0 else 0,
            })
        
        return pd.DataFrame(records)


# Convenience function for quick access
def get_understat_data(fpl_players_df: pd.DataFrame, 
                       teams_df: pd.DataFrame,
                       season: str = "2025") -> pd.DataFrame:
    """
    Quick function to enrich FPL data with Understat metrics.
    
    Args:
        fpl_players_df: FPL players DataFrame
        teams_df: FPL teams DataFrame
        season: Season year
        
    Returns:
        Enriched DataFrame
    """
    service = UnderstatService()
    return service.sync_with_fpl(fpl_players_df, teams_df, season)


if __name__ == "__main__":
    # Test the service
    service = UnderstatService()
    
    # Fetch EPL players
    players = service.get_league_players("2025")
    print(f"Fetched {len(players)} players from Understat")
    
    # Show top 5 by xG
    if players:
        df = service.get_all_stats_summary("2025")
        print("\nTop 5 by xG:")
        print(df.nlargest(5, 'xG')[['player_name', 'team', 'xG', 'goals', 'xG_per_90']])
