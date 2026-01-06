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
    
    def analyze_chip_opportunities_for_horizon(
        self,
        gameweek_plans: List[Dict],
        chips_used: List[str],
        predictions_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze chip opportunities across the optimization horizon.
        
        Args:
            gameweek_plans: List of gameweek plan dicts from optimizer
            chips_used: Chips already used this season
            predictions_df: Optional predictions DataFrame for enhanced analysis
            
        Returns:
            Dict with:
            - chip_recommendations: Ranked list of chip suggestions
            - best_chip: Top recommended chip
            - captain_recommendations: Per-GW captain suggestions
        """
        chip_recommendations = []
        captain_recommendations = []
        
        available_chips = [c for c in self.CHIPS if c not in chips_used]
        
        for gw_plan in gameweek_plans:
            gw = gw_plan.get('gameweek', 0)
            starting_xi = gw_plan.get('starting_xi', [])
            bench = gw_plan.get('bench', [])
            
            # Get DGW/BGW info for this gameweek
            dgw_teams = self.double_gws.get(gw, [])
            bgw_teams = self.blank_gws.get(gw, [])
            is_dgw = len(dgw_teams) > 0
            is_bgw = len(bgw_teams) > 0
            
            # === Captain Analysis ===
            captain_analysis = self._analyze_captain_options(
                starting_xi, gw, dgw_teams, predictions_df
            )
            captain_recommendations.append(captain_analysis)
            
            # === Triple Captain Analysis ===
            if 'triple_captain' in available_chips:
                tc_analysis = self._analyze_triple_captain(
                    starting_xi, gw, dgw_teams, predictions_df
                )
                if tc_analysis:
                    chip_recommendations.append(tc_analysis)
            
            # === Bench Boost Analysis ===
            if 'bench_boost' in available_chips:
                bb_analysis = self._analyze_bench_boost(
                    starting_xi, bench, gw, dgw_teams, predictions_df
                )
                if bb_analysis:
                    chip_recommendations.append(bb_analysis)
            
            # === Free Hit Analysis ===
            if 'free_hit' in available_chips:
                fh_analysis = self._analyze_free_hit(
                    starting_xi, bench, gw, dgw_teams, bgw_teams
                )
                if fh_analysis:
                    chip_recommendations.append(fh_analysis)
        
        # Sort recommendations by estimated gain
        chip_recommendations.sort(key=lambda x: x.get('estimated_gain', 0), reverse=True)
        
        # Determine best chip
        best_chip = chip_recommendations[0] if chip_recommendations else None
        
        return {
            'chip_recommendations': chip_recommendations,
            'best_chip': best_chip,
            'captain_recommendations': captain_recommendations,
            'available_chips': available_chips
        }
    
    def _analyze_captain_options(
        self,
        starting_xi: List[Dict],
        gw: int,
        dgw_teams: List[int],
        predictions_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Analyze captain options for a gameweek."""
        candidates = []
        
        for player in starting_xi:
            player_id = player.get('id')
            player_name = player.get('web_name', 'Unknown')
            team_id = player.get('team')
            
            # Get expected points
            xp_key = f'xp_gw{gw}'
            xp = player.get(xp_key, player.get('expected_points', 0))
            if xp is None:
                xp = player.get('ep_next', 3)
            xp = float(xp) if xp else 3.0
            
            # Check if has DGW
            has_dgw = team_id in dgw_teams if team_id else False
            
            # Get fixtures
            fixtures = self._get_player_fixture_string(team_id, gw)
            
            candidates.append({
                'player_id': player_id,
                'player_name': player_name,
                'expected_points': round(xp, 1),
                'has_dgw': has_dgw,
                'fixtures': fixtures,
                'team_id': team_id
            })
        
        # Sort by xP (DGW players get natural boost from higher xP)
        candidates.sort(key=lambda x: x['expected_points'], reverse=True)
        
        # Top pick
        top_pick = candidates[0] if candidates else None
        alternatives = candidates[1:4] if len(candidates) > 1 else []
        
        # Calculate confidence
        if top_pick and alternatives:
            gap = top_pick['expected_points'] - alternatives[0]['expected_points']
            if gap >= 2.0:
                confidence = 'high'
            elif gap >= 1.0:
                confidence = 'medium'
            else:
                confidence = 'low'
        else:
            confidence = 'medium'
        
        return {
            'gameweek': gw,
            'recommended_captain': top_pick,
            'alternatives': alternatives,
            'confidence': confidence,
            'has_dgw_options': any(c['has_dgw'] for c in candidates[:3])
        }
    
    def _analyze_triple_captain(
        self,
        starting_xi: List[Dict],
        gw: int,
        dgw_teams: List[int],
        predictions_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Analyze Triple Captain value for a gameweek."""
        # Find best captain candidate
        best_captain = None
        best_xp = 0
        
        for player in starting_xi:
            player_id = player.get('id')
            team_id = player.get('team')
            xp_key = f'xp_gw{gw}'
            xp = player.get(xp_key, player.get('expected_points', 0))
            if xp is None:
                xp = player.get('ep_next', 3)
            xp = float(xp) if xp else 3.0
            
            has_dgw = team_id in dgw_teams if team_id else False
            
            if xp > best_xp:
                best_xp = xp
                best_captain = {
                    'player_id': player_id,
                    'player_name': player.get('web_name', 'Unknown'),
                    'expected_points': round(xp, 1),
                    'has_dgw': has_dgw,
                    'fixtures': self._get_player_fixture_string(team_id, gw)
                }
        
        if not best_captain:
            return None
        
        # TC gain = extra captain points (3x instead of 2x = +1x)
        tc_gain = best_xp
        
        # Boost confidence for DGW
        if best_captain['has_dgw']:
            confidence = 'high' if tc_gain >= 8 else 'medium'
            reason = f"{best_captain['player_name']} has DGW{gw} ({best_captain['fixtures']}) - projected {best_xp:.1f} pts"
        else:
            confidence = 'medium' if tc_gain >= 10 else 'low'
            reason = f"{best_captain['player_name']} projected {best_xp:.1f} pts in GW{gw}"
        
        return {
            'chip': 'triple_captain',
            'recommended_gameweek': gw,
            'estimated_gain': round(tc_gain, 1),
            'reason': reason,
            'captain': best_captain,
            'confidence': confidence,
            'is_dgw': best_captain['has_dgw']
        }
    
    def _analyze_bench_boost(
        self,
        starting_xi: List[Dict],
        bench: List[Dict],
        gw: int,
        dgw_teams: List[int],
        predictions_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Analyze Bench Boost value for a gameweek."""
        if not bench:
            return None
        
        total_bench_xp = 0
        dgw_bench_players = 0
        bench_details = []
        
        for player in bench:
            player_id = player.get('id')
            team_id = player.get('team')
            xp_key = f'xp_gw{gw}'
            xp = player.get(xp_key, player.get('expected_points', 0))
            if xp is None:
                xp = player.get('ep_next', 2)
            xp = float(xp) if xp else 2.0
            
            has_dgw = team_id in dgw_teams if team_id else False
            if has_dgw:
                dgw_bench_players += 1
            
            total_bench_xp += xp
            bench_details.append({
                'player_name': player.get('web_name', 'Unknown'),
                'expected_points': round(xp, 1),
                'has_dgw': has_dgw
            })
        
        # BB gain = bench gets full points instead of 0.1x
        bb_gain = total_bench_xp * 0.9
        
        # Confidence based on bench strength and DGW count
        if dgw_bench_players >= 3:
            confidence = 'high'
            reason = f"{dgw_bench_players} bench players have DGW{gw} - total bench xP: {total_bench_xp:.1f}"
        elif dgw_bench_players >= 2 or total_bench_xp >= 15:
            confidence = 'medium'
            reason = f"Strong bench in GW{gw} - total bench xP: {total_bench_xp:.1f}"
        else:
            confidence = 'low'
            reason = f"Bench xP: {total_bench_xp:.1f} in GW{gw}"
        
        return {
            'chip': 'bench_boost',
            'recommended_gameweek': gw,
            'estimated_gain': round(bb_gain, 1),
            'reason': reason,
            'bench_total_xp': round(total_bench_xp, 1),
            'bench_details': bench_details,
            'dgw_bench_count': dgw_bench_players,
            'confidence': confidence,
            'is_dgw': dgw_bench_players > 0
        }
    
    def _analyze_free_hit(
        self,
        starting_xi: List[Dict],
        bench: List[Dict],
        gw: int,
        dgw_teams: List[int],
        bgw_teams: List[int]
    ) -> Optional[Dict]:
        """Analyze Free Hit value for a gameweek."""
        all_players = starting_xi + bench
        
        # Count blanks and DGWs
        blank_count = 0
        dgw_count = 0
        
        for player in all_players:
            team_id = player.get('team')
            if team_id in bgw_teams:
                blank_count += 1
            if team_id in dgw_teams:
                dgw_count += 1
        
        # Free Hit is valuable if:
        # 1. Many squad players blank
        # 2. Big DGW opportunity to maximize DGW players
        
        if blank_count >= 4:
            # High value - many blanks
            estimated_gain = blank_count * 4  # Approximate: 4 pts per player saved
            confidence = 'high'
            reason = f"{blank_count} squad players blank in GW{gw} - Free Hit to field full XI"
            return {
                'chip': 'free_hit',
                'recommended_gameweek': gw,
                'estimated_gain': round(estimated_gain, 1),
                'reason': reason,
                'blank_count': blank_count,
                'dgw_count': dgw_count,
                'confidence': confidence,
                'is_bgw': True,
                'is_dgw': len(dgw_teams) > 5
            }
        elif len(dgw_teams) >= 8 and dgw_count < 8:
            # Big DGW - could field more DGW players
            potential_dgw_gain = (8 - dgw_count) * 2  # ~2 pts per extra DGW player
            estimated_gain = potential_dgw_gain
            confidence = 'medium'
            reason = f"Large DGW{gw} with {len(dgw_teams)} teams - Free Hit to maximize DGW coverage"
            return {
                'chip': 'free_hit',
                'recommended_gameweek': gw,
                'estimated_gain': round(estimated_gain, 1),
                'reason': reason,
                'blank_count': blank_count,
                'dgw_count': dgw_count,
                'confidence': confidence,
                'is_bgw': False,
                'is_dgw': True
            }
        
        return None
    
    def _get_player_fixture_string(self, team_id: int, gw: int) -> str:
        """Get fixture string for a player's team in a specific GW."""
        if not team_id:
            return ""
        
        fixtures = []
        for fix in self.fixtures:
            if fix.get('event') != gw:
                continue
            
            if fix['team_h'] == team_id:
                opp_name = self.teams[self.teams['id'] == fix['team_a']]['short_name'].values
                opp = opp_name[0] if len(opp_name) > 0 else 'UNK'
                fixtures.append(f"{opp}(H)")
            elif fix['team_a'] == team_id:
                opp_name = self.teams[self.teams['id'] == fix['team_h']]['short_name'].values
                opp = opp_name[0] if len(opp_name) > 0 else 'UNK'
                fixtures.append(f"{opp}(A)")
        
        return ", ".join(fixtures) if fixtures else "No fixture"
