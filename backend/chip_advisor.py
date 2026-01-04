"""
Chip Advisor Module for FPL Optimizer

Recommends optimal timing for FPL chips:
- Wildcard: Major squad overhaul
- Free Hit: Single GW with temporary team
- Bench Boost: Double gameweeks with strong bench
- Triple Captain: Double GW or easy fixtures for premium
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from fixture_analyzer import FixtureAnalyzer


class ChipAdvisor:
    """
    Advisor for optimal chip usage timing.
    """
    
    CHIPS = ['wildcard', 'free_hit', 'bench_boost', 'triple_captain']
    
    def __init__(self, 
                 players_df: pd.DataFrame,
                 teams_df: pd.DataFrame,
                 fixtures: List[Dict],
                 current_gw: int):
        """
        Initialize with FPL data.
        
        Args:
            players_df: Player data
            teams_df: Team data
            fixtures: All fixtures
            current_gw: Current gameweek
        """
        self.players = players_df
        self.teams = teams_df
        self.fixtures = fixtures
        self.current_gw = current_gw
        
        self.fixture_analyzer = FixtureAnalyzer(fixtures, teams_df, current_gw)
        
        # Build DGW/BGW map
        self._analyze_special_gameweeks()
    
    def _analyze_special_gameweeks(self):
        """Find double and blank gameweeks for all teams."""
        self.double_gws = {}  # gw -> list of teams with double
        self.blank_gws = {}   # gw -> list of teams with blank
        
        # Count fixtures per team per GW
        gw_fixture_count = {}  # (team_id, gw) -> count
        
        for fix in self.fixtures:
            gw = fix.get('event')
            if gw is None:
                continue
            
            for team_id in [fix['team_h'], fix['team_a']]:
                key = (team_id, gw)
                gw_fixture_count[key] = gw_fixture_count.get(key, 0) + 1
        
        # Find doubles and blanks
        for gw in range(self.current_gw, min(self.current_gw + 15, 39)):
            doubles = []
            blanks = []
            
            for team_id in self.teams['id']:
                count = gw_fixture_count.get((team_id, gw), 0)
                if count >= 2:
                    doubles.append(team_id)
                elif count == 0:
                    blanks.append(team_id)
            
            if doubles:
                self.double_gws[gw] = doubles
            if blanks:
                self.blank_gws[gw] = blanks
    
    def recommend_wildcard(self, current_squad: List[int]) -> Dict:
        """
        Recommend optimal Wildcard timing.
        
        Best when:
        - Before good fixture swing
        - Squad has many injured/suspended players
        - International break (player rest)
        """
        # Find gameweeks with best fixture swings
        best_gw = self.current_gw
        best_reason = "General squad improvement"
        
        # Check if upcoming DGWs warrant waiting
        upcoming_dgws = [gw for gw in self.double_gws if gw > self.current_gw]
        if upcoming_dgws:
            dgw = min(upcoming_dgws)
            if dgw - self.current_gw <= 5:
                best_gw = dgw - 1  # Use WC before DGW
                best_reason = f"Use before DGW{dgw} to maximize double gameweek players"
        
        # Calculate avg fixtures for current squad
        current_avg_fdr = self._calculate_squad_fdr(current_squad)
        
        return {
            "chip": "wildcard",
            "recommended_gw": best_gw,
            "reason": best_reason,
            "current_squad_avg_fdr": round(current_avg_fdr, 2),
            "double_gameweeks_ahead": upcoming_dgws[:3],
            "blank_gameweeks_ahead": list(self.blank_gws.keys())[:3],
            "urgency": "high" if current_avg_fdr > 3.5 else "medium"
        }
    
    def recommend_free_hit(self, current_squad: List[int]) -> Dict:
        """
        Recommend optimal Free Hit timing.
        
        Best for:
        - Blank gameweeks (many teams not playing)
        - Double gameweeks (field best DGW players)
        """
        # Find worst blank gameweek
        best_gw = None
        best_reason = "No clear Free Hit opportunity yet"
        
        # Blank GWs are ideal for Free Hit
        if self.blank_gws:
            gw_impact = {}
            for gw, blanking_teams in self.blank_gws.items():
                # Count how many of current squad are affected
                affected = sum(1 for pid in current_squad 
                              if self._get_player_team(pid) in blanking_teams)
                gw_impact[gw] = affected
            
            if gw_impact:
                worst_gw = max(gw_impact.items(), key=lambda x: x[1])
                if worst_gw[1] >= 3:
                    best_gw = worst_gw[0]
                    best_reason = f"GW{best_gw}: {worst_gw[1]} squad players blank"
        
        # Also consider massive DGWs
        if not best_gw and self.double_gws:
            biggest_dgw = max(self.double_gws.items(), key=lambda x: len(x[1]))
            if len(biggest_dgw[1]) >= 8:
                best_gw = biggest_dgw[0]
                best_reason = f"GW{best_gw}: {len(biggest_dgw[1])} teams have double fixtures"
        
        return {
            "chip": "free_hit",
            "recommended_gw": best_gw or self.current_gw + 10,
            "reason": best_reason,
            "blank_gameweeks": dict(list(self.blank_gws.items())[:5]),
            "double_gameweeks": dict(list(self.double_gws.items())[:5]),
            "urgency": "high" if best_gw and best_gw - self.current_gw <= 3 else "low"
        }
    
    def recommend_bench_boost(self, current_squad: List[int]) -> Dict:
        """
        Recommend optimal Bench Boost timing.
        
        Best when:
        - Double gameweek with strong bench players
        - All 15 players have fixtures
        """
        best_gw = None
        best_reason = "Wait for a double gameweek with strong bench"
        
        if self.double_gws:
            # Find DGW where most of squad has doubles
            for gw, dgw_teams in sorted(self.double_gws.items()):
                squad_dgw_count = sum(1 for pid in current_squad 
                                     if self._get_player_team(pid) in dgw_teams)
                if squad_dgw_count >= 10:
                    best_gw = gw
                    best_reason = f"GW{gw}: {squad_dgw_count} squad players have double fixtures"
                    break
        
        return {
            "chip": "bench_boost",
            "recommended_gw": best_gw or self.current_gw + 10,
            "reason": best_reason,
            "tip": "Ensure bench players are nailed and have good fixtures",
            "urgency": "medium" if best_gw else "low"
        }
    
    def recommend_triple_captain(self, current_squad: List[int]) -> Dict:
        """
        Recommend optimal Triple Captain timing.
        
        Best when:
        - Premium player (Haaland, Salah) has double gameweek
        - Premium has very easy fixtures (FDR 2 or below)
        """
        best_gw = None
        best_player = None
        best_reason = "Wait for premium player's double gameweek"
        
        # Find premium players (cost > 10.0)
        premiums = self.players[self.players['now_cost'] >= 100]
        
        if not premiums.empty and self.double_gws:
            for gw, dgw_teams in sorted(self.double_gws.items()):
                for _, player in premiums.iterrows():
                    if player['team'] in dgw_teams:
                        # Check if this player is in squad or worth captaining
                        best_gw = gw
                        best_player = player['web_name']
                        best_reason = f"GW{gw}: {best_player} has double gameweek"
                        break
                if best_gw:
                    break
        
        return {
            "chip": "triple_captain",
            "recommended_gw": best_gw or self.current_gw + 10,
            "recommended_player": best_player,
            "reason": best_reason,
            "tip": "Target Haaland/Salah in DGW with easy fixtures",
            "urgency": "medium" if best_gw else "low"
        }
    
    def get_all_recommendations(self, current_squad: List[int], 
                                 chips_used: List[str] = []) -> Dict:
        """
        Get recommendations for all available chips.
        
        Args:
            current_squad: Current 15 player IDs
            chips_used: Already used chips
            
        Returns:
            Dict with recommendations for each available chip
        """
        recommendations = {}
        
        if 'wildcard' not in chips_used:
            recommendations['wildcard'] = self.recommend_wildcard(current_squad)
        
        if 'free_hit' not in chips_used:
            recommendations['free_hit'] = self.recommend_free_hit(current_squad)
        
        if 'bench_boost' not in chips_used:
            recommendations['bench_boost'] = self.recommend_bench_boost(current_squad)
        
        if 'triple_captain' not in chips_used:
            recommendations['triple_captain'] = self.recommend_triple_captain(current_squad)
        
        # Add summary
        upcoming_special_gws = set()
        upcoming_special_gws.update(self.double_gws.keys())
        upcoming_special_gws.update(self.blank_gws.keys())
        
        recommendations['summary'] = {
            "next_double_gw": min(self.double_gws.keys()) if self.double_gws else None,
            "next_blank_gw": min(self.blank_gws.keys()) if self.blank_gws else None,
            "chips_remaining": [c for c in self.CHIPS if c not in chips_used],
            "special_gameweeks": sorted(upcoming_special_gws)[:5]
        }
        
        return recommendations
    
    def _calculate_squad_fdr(self, squad: List[int], num_gws: int = 5) -> float:
        """Calculate average FDR for squad over next N gameweeks."""
        if not squad:
            return 3.0
        
        total_fdr = 0
        count = 0
        
        for pid in squad:
            team_id = self._get_player_team(pid)
            if team_id:
                fdr = self.fixture_analyzer.get_avg_fdr(team_id, num_gws)
                total_fdr += fdr
                count += 1
        
        return total_fdr / count if count > 0 else 3.0
    
    def _get_player_team(self, player_id: int) -> Optional[int]:
        """Get team ID for a player."""
        player = self.players[self.players['id'] == player_id]
        if player.empty:
            return None
        return player.iloc[0]['team']
