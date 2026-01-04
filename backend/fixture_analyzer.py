"""
Fixture Analyzer Module for FPL Optimizer

Analyzes upcoming fixtures and provides:
- Fixture Difficulty Rating (FDR) for each player
- Average difficulty over N gameweeks
- Blank/Double gameweek detection
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd


class FixtureAnalyzer:
    """
    Analyzes fixtures and provides difficulty ratings for players.
    """
    
    # FDR color mapping
    FDR_COLORS = {
        1: "green",   # Very easy
        2: "green",   # Easy
        3: "yellow",  # Medium
        4: "red",     # Hard
        5: "red"      # Very hard
    }
    
    def __init__(self, fixtures: List[Dict], teams_df: pd.DataFrame, current_gw: int):
        """
        Initialize with fixtures and team data.
        
        Args:
            fixtures: List of fixture dicts from FPL API
            teams_df: DataFrame of teams
            current_gw: Current gameweek number
        """
        self.fixtures = fixtures
        self.teams = teams_df
        self.current_gw = current_gw
        
        # Build team name lookup
        self.team_names = {row['id']: row['short_name'] for _, row in teams_df.iterrows()}
        self.team_strength = {row['id']: row.get('strength', 3) for _, row in teams_df.iterrows()}
        
        # Build fixture map: team_id -> list of upcoming fixtures
        self.fixture_map = self._build_fixture_map()
    
    def _build_fixture_map(self) -> Dict[int, List[Dict]]:
        """Build a map of team_id -> upcoming fixtures."""
        fixture_map = {}
        
        for fix in self.fixtures:
            gw = fix.get('event')
            if gw is None or gw < self.current_gw:
                continue
                
            home_team = fix['team_h']
            away_team = fix['team_a']
            
            # Home fixture
            if home_team not in fixture_map:
                fixture_map[home_team] = []
            fixture_map[home_team].append({
                'gameweek': gw,
                'opponent': away_team,
                'opponent_name': self.team_names.get(away_team, 'UNK'),
                'is_home': True,
                'fdr': fix.get('team_h_difficulty', 3)
            })
            
            # Away fixture
            if away_team not in fixture_map:
                fixture_map[away_team] = []
            fixture_map[away_team].append({
                'gameweek': gw,
                'opponent': home_team,
                'opponent_name': self.team_names.get(home_team, 'UNK'),
                'is_home': False,
                'fdr': fix.get('team_a_difficulty', 3)
            })
        
        # Sort by gameweek
        for team_id in fixture_map:
            fixture_map[team_id].sort(key=lambda x: x['gameweek'])
        
        return fixture_map
    
    def get_player_fixtures(self, team_id: int, num_gameweeks: int = 6) -> List[Dict]:
        """
        Get upcoming fixtures for a player's team.
        
        Args:
            team_id: Player's team ID
            num_gameweeks: Number of upcoming GWs to include
            
        Returns:
            List of fixture dicts with opponent, FDR, home/away
        """
        team_fixtures = self.fixture_map.get(team_id, [])
        
        # Filter to next N gameweeks
        upcoming = [
            f for f in team_fixtures 
            if f['gameweek'] < self.current_gw + num_gameweeks
        ]
        
        return upcoming[:num_gameweeks]
    
    def get_fixture_string(self, team_id: int, num_gameweeks: int = 5) -> str:
        """
        Get a compact fixture string for display.
        
        Example: "MCI(A)5 → LEI(H)2 → EVE(A)3"
        """
        fixtures = self.get_player_fixtures(team_id, num_gameweeks)
        
        parts = []
        for fix in fixtures:
            venue = "H" if fix['is_home'] else "A"
            parts.append(f"{fix['opponent_name']}({venue}){fix['fdr']}")
        
        return " → ".join(parts)
    
    def get_avg_fdr(self, team_id: int, num_gameweeks: int = 5) -> float:
        """
        Calculate average FDR over next N gameweeks.
        
        Lower is easier.
        """
        fixtures = self.get_player_fixtures(team_id, num_gameweeks)
        
        if not fixtures:
            return 3.0  # Default medium difficulty
        
        total_fdr = sum(f['fdr'] for f in fixtures)
        return total_fdr / len(fixtures)
    
    def find_blank_gameweeks(self, team_id: int) -> List[int]:
        """
        Find gameweeks where a team has no fixture (blank GW).
        """
        team_fixtures = self.fixture_map.get(team_id, [])
        fixture_gws = {f['gameweek'] for f in team_fixtures}
        
        blanks = []
        for gw in range(self.current_gw, min(self.current_gw + 10, 39)):
            if gw not in fixture_gws:
                blanks.append(gw)
        
        return blanks
    
    def find_double_gameweeks(self, team_id: int) -> List[int]:
        """
        Find gameweeks where a team has multiple fixtures (double GW).
        """
        team_fixtures = self.fixture_map.get(team_id, [])
        
        gw_counts = {}
        for fix in team_fixtures:
            gw = fix['gameweek']
            gw_counts[gw] = gw_counts.get(gw, 0) + 1
        
        doubles = [gw for gw, count in gw_counts.items() if count > 1]
        return sorted(doubles)
    
    def get_fixture_ticker(self, team_id: int, num_gameweeks: int = 5) -> List[Dict]:
        """
        Get fixture data formatted for frontend ticker display.
        
        Returns list of dicts with opponent, fdr, color, venue.
        """
        fixtures = self.get_player_fixtures(team_id, num_gameweeks)
        
        ticker = []
        for fix in fixtures:
            fdr = fix['fdr']
            ticker.append({
                'gw': fix['gameweek'],
                'opponent': fix['opponent_name'],
                'venue': 'H' if fix['is_home'] else 'A',
                'fdr': fdr,
                'color': self.FDR_COLORS.get(fdr, 'yellow'),
                'display': f"{fix['opponent_name']}({('H' if fix['is_home'] else 'A')})"
            })
        
        return ticker
    
    def enrich_players_with_fixtures(self, players_df: pd.DataFrame, 
                                      num_gameweeks: int = 5) -> pd.DataFrame:
        """
        Add fixture information to player DataFrame.
        
        Adds columns:
        - fixture_ticker: List of upcoming fixtures
        - avg_fdr: Average FDR over N gameweeks
        - fixture_string: Compact display string
        """
        df = players_df.copy()
        
        df['fixture_ticker'] = df['team'].apply(
            lambda t: self.get_fixture_ticker(t, num_gameweeks)
        )
        df['avg_fdr'] = df['team'].apply(
            lambda t: round(self.get_avg_fdr(t, num_gameweeks), 2)
        )
        df['fixture_string'] = df['team'].apply(
            lambda t: self.get_fixture_string(t, num_gameweeks)
        )
        df['blank_gws'] = df['team'].apply(
            lambda t: self.find_blank_gameweeks(t)
        )
        df['double_gws'] = df['team'].apply(
            lambda t: self.find_double_gameweeks(t)
        )
        
        return df
