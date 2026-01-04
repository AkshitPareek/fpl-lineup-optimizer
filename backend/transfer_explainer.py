"""
Transfer Explainer Module for FPL 2025/26

Generates human-readable, LLM-style explanations for transfer recommendations
using player metrics (xG, xA, form), fixture analysis, and comparative reasoning.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class PlayerComparison:
    """Comparison between transferred in and out players."""
    player_in: Dict[str, Any]
    player_out: Dict[str, Any]
    reasons: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]


@dataclass 
class TransferExplanation:
    """Complete explanation for a transfer."""
    gameweek: int
    summary: str
    detailed_analysis: str
    key_metrics: Dict[str, Any]
    fixture_analysis: str
    risk_assessment: str


class TransferExplainer:
    """
    Generates human-readable explanations for transfer recommendations.
    
    Creates narratives explaining WHY players are recommended, using:
    - xG/xA trends and comparisons
    - Fixture difficulty analysis
    - Form vs expected performance gaps
    - Team and position context
    """
    
    def __init__(self, 
                 players_df: pd.DataFrame,
                 teams_df: pd.DataFrame,
                 fixtures: List[Dict]):
        """
        Initialize explainer with FPL data.
        
        Args:
            players_df: Player data with stats
            teams_df: Team data
            fixtures: Fixture list
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        self.fixtures = fixtures
        
        # Build lookup maps
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        self.team_short_names = dict(zip(self.teams['id'], self.teams['short_name']))
        
        # Build fixture difficulty map (team_id -> list of (gw, opponent, difficulty, is_home))
        self._build_fixture_map()
        
    def _build_fixture_map(self):
        """Build fixture lookup by team and gameweek."""
        self.fixture_map = {}
        
        for fix in self.fixtures:
            gw = fix.get('event')
            if gw is None:
                continue
            
            # Home team
            home_id = fix['team_h']
            away_id = fix['team_a']
            
            if home_id not in self.fixture_map:
                self.fixture_map[home_id] = []
            self.fixture_map[home_id].append({
                'gameweek': gw,
                'opponent': away_id,
                'opponent_name': self.team_short_names.get(away_id, 'UNK'),
                'difficulty': fix.get('team_h_difficulty', 3),
                'is_home': True
            })
            
            if away_id not in self.fixture_map:
                self.fixture_map[away_id] = []
            self.fixture_map[away_id].append({
                'gameweek': gw,
                'opponent': home_id,
                'opponent_name': self.team_short_names.get(home_id, 'UNK'),
                'difficulty': fix.get('team_a_difficulty', 3),
                'is_home': False
            })
    
    def _get_player_stats(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get player stats by ID."""
        player = self.players[self.players['id'] == player_id]
        if player.empty:
            return None
        return player.iloc[0].to_dict()
    
    def _get_upcoming_fixtures(self, team_id: int, gameweeks: int = 5, from_gw: int = 1) -> List[Dict]:
        """Get upcoming fixtures for a team."""
        if team_id not in self.fixture_map:
            return []
        
        fixtures = [f for f in self.fixture_map[team_id] 
                   if from_gw <= f['gameweek'] < from_gw + gameweeks]
        return sorted(fixtures, key=lambda x: x['gameweek'])
    
    def _format_fixtures(self, fixtures: List[Dict]) -> str:
        """Format fixtures in readable string."""
        if not fixtures:
            return "No fixtures available"
        
        parts = []
        for fix in fixtures:
            opp = fix['opponent_name']
            home_away = '(H)' if fix['is_home'] else '(A)'
            fdr = fix['difficulty']
            fdr_emoji = '🟢' if fdr <= 2 else ('🟡' if fdr == 3 else ('🟠' if fdr == 4 else '🔴'))
            parts.append(f"{opp}{home_away}{fdr_emoji}")
        
        return ' → '.join(parts)
    
    def _calculate_xg_trend(self, player: Dict) -> Dict[str, Any]:
        """Calculate xG trend indicators."""
        minutes = float(player.get('minutes', 0) or 0)
        expected_goals = float(player.get('expected_goals', 0) or 0)
        goals_scored = float(player.get('goals_scored', 0) or 0)
        
        xg_per_90 = (expected_goals / minutes * 90) if minutes > 0 else 0
        goals_per_90 = (goals_scored / minutes * 90) if minutes > 0 else 0
        
        # Over/underperformance
        if expected_goals > 0:
            performance_ratio = goals_scored / expected_goals
        else:
            performance_ratio = 1.0
        
        return {
            'xg_per_90': round(xg_per_90, 3),
            'goals_per_90': round(goals_per_90, 3),
            'performance_ratio': round(performance_ratio, 2),
            'overperforming': performance_ratio > 1.15,
            'underperforming': performance_ratio < 0.85
        }
    
    def _calculate_xa_trend(self, player: Dict) -> Dict[str, Any]:
        """Calculate xA trend indicators."""
        minutes = float(player.get('minutes', 0) or 0)
        expected_assists = float(player.get('expected_assists', 0) or 0)
        assists = float(player.get('assists', 0) or 0)
        
        xa_per_90 = (expected_assists / minutes * 90) if minutes > 0 else 0
        assists_per_90 = (assists / minutes * 90) if minutes > 0 else 0
        
        if expected_assists > 0:
            performance_ratio = assists / expected_assists
        else:
            performance_ratio = 1.0
        
        return {
            'xa_per_90': round(xa_per_90, 3),
            'assists_per_90': round(assists_per_90, 3),
            'performance_ratio': round(performance_ratio, 2),
            'overperforming': performance_ratio > 1.15,
            'underperforming': performance_ratio < 0.85
        }
    
    def explain_transfer_out(self, player_id: int, current_gw: int, horizon: int = 5) -> Dict[str, Any]:
        """
        Generate explanation for why a player should be transferred OUT.
        
        Args:
            player_id: ID of player being transferred out
            current_gw: Current gameweek
            horizon: How many GWs to consider
            
        Returns:
            Dict with summary and reasons
        """
        player = self._get_player_stats(player_id)
        if not player:
            return {"summary": "Player not found", "reasons": []}
        
        name = player['web_name']
        team_name = self.team_names.get(player['team'], 'Unknown')
        position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['element_type'], 'UNK')
        cost = player['now_cost'] / 10.0
        
        # Get metrics
        xg_trend = self._calculate_xg_trend(player)
        xa_trend = self._calculate_xa_trend(player)
        form = float(player.get('form', 0) or 0)
        
        # Get fixtures
        fixtures = self._get_upcoming_fixtures(player['team'], horizon, current_gw)
        fixture_str = self._format_fixtures(fixtures)
        avg_fdr = sum(f['difficulty'] for f in fixtures) / len(fixtures) if fixtures else 3
        
        reasons = []
        
        # Bad form
        if form < 3.0:
            reasons.append(f"Poor form ({form:.1f} pts/game)")
        
        # Tough fixtures
        if avg_fdr >= 3.5:
            reasons.append(f"Tough upcoming fixtures (avg FDR: {avg_fdr:.1f})")
        
        # Overperforming xG (regression risk)
        if xg_trend['overperforming']:
            reasons.append(f"Overperforming xG - regression likely")
        
        # Low xG output
        if xg_trend['xg_per_90'] < 0.2 and player['element_type'] in [3, 4]:  # MID/FWD
            reasons.append(f"Low xG/90 ({xg_trend['xg_per_90']:.2f})")
        
        # Injury concern
        chance = player.get('chance_of_playing_next_round')
        if chance is not None and not pd.isna(chance) and int(chance) < 75:
            reasons.append(f"Injury doubt ({int(chance)}% available)")
        
        # Low minutes
        minutes = float(player.get('minutes', 0) or 0)
        if minutes > 0 and minutes < 500:
            reasons.append("Rotation risk - limited minutes")
        
        # No reasons = maybe shouldn't sell
        if not reasons:
            reasons.append("Better options available elsewhere")
        
        summary = f"**{name}** ({team_name} {position}, £{cost}m) - " + ", ".join(reasons[:2])
        
        return {
            "summary": summary,
            "reasons": reasons,
            "fixtures": fixture_str,
            "form": form,
            "avg_fdr": avg_fdr
        }
    
    def explain_transfer_in(self, 
                             player_id: int,
                             current_gw: int,
                             horizon: int = 5,
                             replacing_id: Optional[int] = None) -> TransferExplanation:
        """
        Generate detailed explanation for a transfer IN recommendation.
        
        Args:
            player_id: ID of player being transferred in
            current_gw: Current gameweek
            horizon: How many GWs to consider
            replacing_id: ID of player being replaced (optional)
            
        Returns:
            TransferExplanation with full narrative
        """
        player = self._get_player_stats(player_id)
        if not player:
            return TransferExplanation(
                gameweek=current_gw,
                summary="Player not found",
                detailed_analysis="",
                key_metrics={},
                fixture_analysis="",
                risk_assessment=""
            )
        
        name = player['web_name']
        team_name = self.team_names.get(player['team'], 'Unknown')
        position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['element_type'], 'UNK')
        cost = player['now_cost'] / 10.0
        
        # Get metrics
        xg_trend = self._calculate_xg_trend(player)
        xa_trend = self._calculate_xa_trend(player)
        form = float(player.get('form', 0) or 0)
        ict = float(player.get('ict_index', 0) or 0)
        selected_by = float(player.get('selected_by_percent', 0) or 0)
        
        # Get fixtures
        fixtures = self._get_upcoming_fixtures(player['team'], horizon, current_gw)
        fixture_str = self._format_fixtures(fixtures)
        avg_fdr = sum(f['difficulty'] for f in fixtures) / len(fixtures) if fixtures else 3
        
        # Build summary
        summary_parts = [f"**{name}** ({team_name} {position}, £{cost}m)"]
        
        reasons = []
        
        # Fixture-based reasoning
        if avg_fdr < 2.5:
            reasons.append(f"excellent upcoming fixtures (avg FDR: {avg_fdr:.1f})")
        elif avg_fdr < 3:
            reasons.append(f"favorable fixtures (avg FDR: {avg_fdr:.1f})")
        
        # Form-based reasoning
        if form >= 6.0:
            reasons.append(f"outstanding form ({form:.1f})")
        elif form >= 4.5:
            reasons.append(f"good form ({form:.1f})")
        
        # xG/xA reasoning
        if xg_trend['xg_per_90'] > 0.4:
            reasons.append(f"high xG/90 ({xg_trend['xg_per_90']:.2f})")
        if xa_trend['xa_per_90'] > 0.3:
            reasons.append(f"high xA/90 ({xa_trend['xa_per_90']:.2f})")
        
        # Underperformance = upside
        if xg_trend['underperforming']:
            reasons.append("currently underperforming xG (likely to regress upward)")
        
        # ICT reasoning
        if ict > 100:
            reasons.append(f"strong ICT Index ({ict:.1f})")
        
        # Ownership differential
        if selected_by < 10:
            reasons.append(f"low ownership differential ({selected_by:.1f}%)")
        elif selected_by > 40:
            reasons.append(f"highly owned template pick ({selected_by:.1f}%)")
        
        if reasons:
            summary = f"{summary_parts[0]} is recommended because of " + ", ".join(reasons[:3]) + "."
        else:
            summary = f"{summary_parts[0]} is a solid option for the upcoming fixtures."
        
        # Detailed analysis
        detailed = f"""
### {name} Analysis

**Attacking Output:**
- xG per 90: {xg_trend['xg_per_90']:.3f} | Goals per 90: {xg_trend['goals_per_90']:.3f}
- xA per 90: {xa_trend['xa_per_90']:.3f} | Assists per 90: {xa_trend['assists_per_90']:.3f}
- ICT Index: {ict:.1f}

**Form & Value:**
- Current Form: {form:.1f}
- Cost: £{cost}m
- Selected By: {selected_by:.1f}%

**Fixture Outlook:**
{fixture_str}
Average FDR: {avg_fdr:.2f}
        """.strip()
        
        # Risk assessment
        risks = []
        if xg_trend['overperforming']:
            risks.append("Currently overperforming xG - may regress")
        chance_of_playing = player.get('chance_of_playing_next_round')
        if chance_of_playing is not None and not pd.isna(chance_of_playing) and int(chance_of_playing) < 75:
            risks.append(f"Injury concern ({int(chance_of_playing)}% chance of playing)")
        if avg_fdr > 3.5:
            risks.append("Tough upcoming fixtures")
        if float(player.get('minutes', 0) or 0) < 500:
            risks.append("Limited minutes this season - rotation risk")
        
        risk_str = " | ".join(risks) if risks else "No significant risks identified."
        
        # Comparison with replaced player
        if replacing_id:
            old_player = self._get_player_stats(replacing_id)
            if old_player:
                old_fixtures = self._get_upcoming_fixtures(old_player['team'], horizon, current_gw)
                old_avg_fdr = sum(f['difficulty'] for f in old_fixtures) / len(old_fixtures) if old_fixtures else 3
                
                comparison = f"\n\n**Why replace {old_player['web_name']}?**\n"
                if avg_fdr < old_avg_fdr - 0.5:
                    comparison += f"- Better fixtures ({avg_fdr:.2f} vs {old_avg_fdr:.2f} avg FDR)\n"
                if form > float(old_player.get('form', 0) or 0) + 1:
                    comparison += f"- Better form ({form:.1f} vs {old_player.get('form', 0)})\n"
                if xg_trend['xg_per_90'] > self._calculate_xg_trend(old_player)['xg_per_90'] * 1.2:
                    comparison += f"- Higher xG output\n"
                
                detailed += comparison
        
        return TransferExplanation(
            gameweek=current_gw,
            summary=summary,
            detailed_analysis=detailed,
            key_metrics={
                'xg_per_90': xg_trend['xg_per_90'],
                'xa_per_90': xa_trend['xa_per_90'],
                'form': form,
                'ict_index': ict,
                'avg_fdr': avg_fdr,
                'selected_by': selected_by
            },
            fixture_analysis=fixture_str,
            risk_assessment=risk_str
        )
    
    def explain_multi_transfer(self,
                                transfers_in: List[Dict],
                                transfers_out: List[Dict],
                                current_gw: int,
                                horizon: int = 5) -> str:
        """
        Generate explanation for multiple transfers.
        
        Args:
            transfers_in: List of players transferred in
            transfers_out: List of players transferred out
            current_gw: Current gameweek
            horizon: Planning horizon
            
        Returns:
            Formatted explanation string
        """
        if not transfers_in:
            return "No transfers recommended this gameweek."
        
        lines = [f"## Gameweek {current_gw} Transfer Analysis\n"]
        
        # Match transfers by position
        # Create a mapping of position -> transfers out
        out_by_position = {}
        for t_out in transfers_out:
            pos = t_out.get('element_type', 0)
            if pos not in out_by_position:
                out_by_position[pos] = []
            out_by_position[pos].append(t_out)
        
        for i, t_in in enumerate(transfers_in):
            player_id = t_in.get('id')
            pos = t_in.get('element_type', 0)
            in_xp = t_in.get('expected_points', 0)
            
            # Find matching position transfer out
            replacing_id = None
            out_xp = 0
            if pos in out_by_position and out_by_position[pos]:
                t_out = out_by_position[pos].pop(0)
                replacing_id = t_out['id']
                # Get expected points for player being sold
                out_player = self._get_player_stats(replacing_id)
                if out_player:
                    form = float(out_player.get('form', 0) or 0)
                    out_xp = form  # Estimate xP based on recent form
            
            xp_gain = in_xp - out_xp
            
            lines.append(f"### Transfer {i+1}")
            
            # Explain OUT first
            if replacing_id:
                out_explanation = self.explain_transfer_out(
                    player_id=replacing_id,
                    current_gw=current_gw,
                    horizon=horizon
                )
                lines.append(f"**SELL:** {out_explanation['summary']}")
                lines.append(f"  └ Fixtures: {out_explanation['fixtures']}")
                lines.append(f"  └ Expected: {out_xp:.1f} xP")
            
            # Explain IN
            in_explanation = self.explain_transfer_in(
                player_id=player_id,
                current_gw=current_gw,
                horizon=horizon,
                replacing_id=replacing_id
            )
            lines.append(f"\n**BUY:** {in_explanation.summary}")
            lines.append(f"  └ Fixtures: {in_explanation.fixture_analysis}")
            lines.append(f"  └ Expected: {in_xp:.1f} xP")
            lines.append(f"  └ Risks: {in_explanation.risk_assessment}")
            
            # xP Gain
            gain_color = "🟢" if xp_gain > 0 else "🔴"
            lines.append(f"\n{gain_color} **xP Gain: {xp_gain:+.1f}**")
            lines.append("")
        
        # Cost analysis
        total_in_cost = sum(t.get('cost', 0) for t in transfers_in)
        total_out_cost = sum(t.get('cost', 0) for t in transfers_out)
        net = total_out_cost - total_in_cost
        
        # Total xP gain calculation
        total_in_xp = sum(t.get('expected_points', 0) for t in transfers_in)
        
        lines.append(f"---")
        lines.append(f"**Net Transfer Cost:** £{net:+.1f}m")
        lines.append(f"**Total Expected Points (new players):** {total_in_xp:.1f} xP")
        
        return "\n".join(lines)
    
    def generate_plan_explanation(self,
                                   gameweek_plans: List[Dict],
                                   current_gw: int) -> str:
        """
        Generate complete explanation for a multi-gameweek plan.
        
        Args:
            gameweek_plans: List of gameweek plan dicts from optimizer
            current_gw: Starting gameweek
            
        Returns:
            Full markdown-formatted explanation
        """
        lines = ["# Multi-Gameweek Transfer Plan\n"]
        
        total_xp = 0
        total_hits = 0
        
        for plan in gameweek_plans:
            gw = plan['gameweek']
            xp = plan['expected_points']
            total_xp += xp
            
            transfers = plan.get('transfers', {})
            hits = transfers.get('hits', 0)
            total_hits += hits * 4
            
            lines.append(f"## Gameweek {gw}")
            lines.append(f"**Expected Points:** {xp:.1f}")
            
            if transfers.get('in'):
                explanation = self.explain_multi_transfer(
                    transfers_in=transfers['in'],
                    transfers_out=transfers['out'],
                    current_gw=gw,
                    horizon=5
                )
                lines.append(explanation)
            else:
                lines.append("*No transfers*")
            
            lines.append("")
        
        lines.append("---")
        lines.append(f"**Total Expected Points:** {total_xp:.1f}")
        lines.append(f"**Total Hit Cost:** {total_hits} points")
        lines.append(f"**Net Expected Points:** {total_xp - total_hits:.1f}")
        
        return "\n".join(lines)
