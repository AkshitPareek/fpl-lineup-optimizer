"""
Point Predictor Module for FPL 2025/26 Season

Calculates expected points using:
- xG (expected goals)
- xA (expected assists)
- xGI (expected goal involvement)
- ICT Index components
- CBIT/CBIRT defensive bonus scoring (2025/26 rules)
- Fixture difficulty rating (FDR)
- Clean sheet probability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PointPrediction:
    """Container for multi-gameweek point predictions."""
    player_id: int
    player_name: str
    position: str
    team: str
    gameweek_predictions: Dict[int, float]  # GW -> expected points
    total_expected: float
    breakdown: Dict[str, float]  # Component breakdown


class PointPredictor:
    """
    Advanced expected points calculator for FPL 2025/26.
    
    Incorporates:
    - Expected metrics (xG, xA, xGI)
    - ICT Index for form assessment
    - CBIT/CBIRT defensive bonus (NEW for 2025/26)
    - Fixture difficulty and opponent strength
    - Clean sheet probability modeling
    """
    
    # FPL Scoring Rules
    POINTS_GOAL = {1: 6, 2: 6, 3: 5, 4: 4}  # GK, DEF, MID, FWD
    POINTS_ASSIST = 3
    POINTS_CLEAN_SHEET = {1: 4, 2: 4, 3: 1, 4: 0}
    POINTS_SAVE = 1 / 3  # 1 point per 3 saves
    POINTS_APPEARANCE = 2  # For 60+ mins
    POINTS_BONUS_AVG = 0.75  # Expected bonus per game on average
    
    # 2025/26 CBIT/CBIRT Bonus Rules
    CBIT_THRESHOLD_DEF = 10  # Defenders: 10+ CBIT = 2 pts
    CBIRT_THRESHOLD_MID_FWD = 12  # Mid/Fwd: 12+ CBIRT = 2 pts
    POINTS_DEFENSIVE_BONUS = 2
    
    def __init__(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, 
                 fixtures: List[Dict], current_gameweek: int):
        """
        Initialize predictor with FPL data.
        
        Args:
            players_df: DataFrame of all players (from bootstrap-static/elements)
            teams_df: DataFrame of all teams
            fixtures: List of fixture dicts from FPL API
            current_gameweek: Current or next gameweek number
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        self.fixtures = fixtures
        self.current_gw = current_gameweek
        
        # Precompute team strengths
        self._compute_team_strengths()
        
        # Parse fixture data into lookup structure
        self._build_fixture_map()
        
    def _compute_team_strengths(self):
        """Compute attack and defense strength ratings for each team."""
        # Use FPL's built-in strength ratings
        self.team_attack_home = dict(zip(
            self.teams['id'], 
            self.teams['strength_attack_home']
        ))
        self.team_attack_away = dict(zip(
            self.teams['id'],
            self.teams['strength_attack_away']
        ))
        self.team_defense_home = dict(zip(
            self.teams['id'],
            self.teams['strength_defence_home']  # Note: FPL uses British spelling
        ))
        self.team_defense_away = dict(zip(
            self.teams['id'],
            self.teams['strength_defence_away']
        ))
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        
    def _build_fixture_map(self):
        """Build lookup structure for fixtures by team and gameweek."""
        self.fixture_map = {}  # (team_id, gw) -> list of fixture info
        
        for fixture in self.fixtures:
            gw = fixture.get('event')
            if gw is None:
                continue
                
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            
            # Home team fixture
            key_home = (home_team, gw)
            if key_home not in self.fixture_map:
                self.fixture_map[key_home] = []
            self.fixture_map[key_home].append({
                'opponent': away_team,
                'is_home': True,
                'difficulty': fixture.get('team_h_difficulty', 3)
            })
            
            # Away team fixture
            key_away = (away_team, gw)
            if key_away not in self.fixture_map:
                self.fixture_map[key_away] = []
            self.fixture_map[key_away].append({
                'opponent': home_team,
                'is_home': False,
                'difficulty': fixture.get('team_a_difficulty', 3)
            })
    
    def get_fixtures(self, team_id: int, gameweeks: int = 5) -> List[Dict]:
        """Get upcoming fixtures for a team."""
        fixtures = []
        for gw in range(self.current_gw, self.current_gw + gameweeks):
            key = (team_id, gw)
            if key in self.fixture_map:
                for fix in self.fixture_map[key]:
                    fix['gameweek'] = gw
                    fixtures.append(fix)
        return fixtures
    
    # Minimum minutes for reliable per-90 calculations (3 full games)
    MIN_MINUTES_FOR_RATES = 270
    
    # Position-based baseline xG rates (per 90 mins) for limited data
    BASELINE_XG = {1: 0.0, 2: 0.03, 3: 0.08, 4: 0.35}
    BASELINE_XA = {1: 0.0, 2: 0.04, 3: 0.08, 4: 0.12}
    
    # Calibration factor based on historical FPL prediction accuracy
    CALIBRATION_FACTOR = 0.85
    
    def calculate_xg_points(self, player: pd.Series, gameweek: int) -> float:
        """
        Calculate expected goal points for a player.
        
        Uses player's xG per 90 adjusted for opponent defense strength.
        Requires minimum sample size for reliable rates.
        """
        element_type = player['element_type']
        
        # Get base xG rate with minimum sample size requirement
        minutes = max(float(player.get('minutes', 0) or 0), 1)
        expected_goals_season = float(player.get('expected_goals', 0) or 0)
        
        if minutes >= self.MIN_MINUTES_FOR_RATES:
            xg_per_90 = (expected_goals_season / minutes) * 90
        else:
            # Use position-based baseline for limited data
            xg_per_90 = self.BASELINE_XG.get(element_type, 0.1)
        
        # Get fixture info
        team_id = player['team']
        fixtures = self.fixture_map.get((team_id, gameweek), [])
        
        if not fixtures:
            return xg_per_90 * self.POINTS_GOAL.get(element_type, 4) * 0.8  # Default estimate
        
        total_xg = 0
        for fix in fixtures:
            opponent = fix['opponent']
            is_home = fix['is_home']
            
            # Adjust xG based on opponent defense (higher strength = fewer goals expected)
            if is_home:
                defense_strength = self.team_defense_away.get(opponent, 1000) / 1000
            else:
                defense_strength = self.team_defense_home.get(opponent, 1000) / 1000
            
            # Dampened fixture multiplier (was 0.5-1.5, now 0.85-1.15)
            multiplier = 1.0 + (1.0 - defense_strength) * 0.15
            adjusted_xg = xg_per_90 * max(0.85, min(1.15, multiplier))
            total_xg += adjusted_xg
        
        goal_points = total_xg * self.POINTS_GOAL.get(element_type, 4)
        # Cap at reasonable maximum (~1.5 goals worth)
        return min(goal_points, 9.0)
    
    def calculate_xa_points(self, player: pd.Series, gameweek: int) -> float:
        """Calculate expected assist points with minimum sample size."""
        element_type = player['element_type']
        minutes = max(float(player.get('minutes', 0) or 0), 1)
        expected_assists_season = float(player.get('expected_assists', 0) or 0)
        
        if minutes >= self.MIN_MINUTES_FOR_RATES:
            xa_per_90 = (expected_assists_season / minutes) * 90
        else:
            # Use position-based baseline for limited data
            xa_per_90 = self.BASELINE_XA.get(element_type, 0.08)
        
        team_id = player['team']
        fixtures = self.fixture_map.get((team_id, gameweek), [])
        
        if not fixtures:
            return xa_per_90 * self.POINTS_ASSIST * 0.8
        
        total_xa = 0
        for fix in fixtures:
            opponent = fix['opponent']
            is_home = fix['is_home']
            
            if is_home:
                defense_strength = self.team_defense_away.get(opponent, 1000) / 1000
            else:
                defense_strength = self.team_defense_home.get(opponent, 1000) / 1000
            
            # Dampened fixture multiplier
            multiplier = 1.0 + (1.0 - defense_strength) * 0.15
            total_xa += xa_per_90 * max(0.85, min(1.15, multiplier))
        
        assist_points = total_xa * self.POINTS_ASSIST
        # Cap at reasonable maximum (~2 assists worth)
        return min(assist_points, 6.0)
    
    def calculate_clean_sheet_prob(self, player: pd.Series, gameweek: int) -> float:
        """
        Estimate clean sheet probability.
        
        Uses team defensive strength vs opponent attack strength.
        """
        element_type = player['element_type']
        if self.POINTS_CLEAN_SHEET.get(element_type, 0) == 0:
            return 0  # Forwards don't get CS points
        
        team_id = player['team']
        fixtures = self.fixture_map.get((team_id, gameweek), [])
        
        if not fixtures:
            return 0.25 * self.POINTS_CLEAN_SHEET.get(element_type, 0)  # Base 25% CS prob
        
        total_cs_prob = 0
        for fix in fixtures:
            opponent = fix['opponent']
            is_home = fix['is_home']
            
            # Team's defensive strength
            if is_home:
                own_defense = self.team_defense_home.get(team_id, 1000) / 1000
                opp_attack = self.team_attack_away.get(opponent, 1000) / 1000
            else:
                own_defense = self.team_defense_away.get(team_id, 1000) / 1000
                opp_attack = self.team_attack_home.get(opponent, 1000) / 1000
            
            # Higher defense + lower opponent attack = higher CS probability
            # Base probability around 25%, adjust by strength differential
            strength_diff = own_defense - opp_attack
            cs_prob = 0.25 + (strength_diff * 0.3)
            cs_prob = max(0.05, min(0.6, cs_prob))  # Clamp between 5% and 60%
            total_cs_prob += cs_prob
        
        return total_cs_prob * self.POINTS_CLEAN_SHEET.get(element_type, 0)
    
    def calculate_cbit_bonus(self, player: pd.Series, gameweek: int) -> float:
        """
        Calculate 2025/26 CBIT/CBIRT defensive bonus points.
        
        Rules:
        - Defenders: 2 pts for 10+ CBIT (Clearances, Blocks, Interceptions, Tackles)
        - Midfielders/Forwards: 2 pts for 12+ CBIRT (CBIT + Ball Recoveries)
        """
        element_type = player['element_type']
        
        # Use ICT 'influence' as proxy for defensive actions
        # 'influence' captures tackles won, interceptions, clearances in its calculation
        influence = float(player.get('influence', 0) or 0)
        minutes = max(float(player.get('minutes', 0) or 0), 1)
        
        # Games played (approximate)
        games = len(self.fixtures) if self.fixtures else 1
        appearances = minutes / 90
        
        # Influence per game as proxy for CBIT/CBIRT potential
        influence_per_game = influence / max(appearances, 1)
        
        # Estimate probability of reaching CBIT/CBIRT threshold
        if element_type == 2:  # Defender
            # Higher influence = more likely to reach 10 CBIT
            # Influence around 40+ per game suggests regular defensive actions
            threshold_prob = min(1.0, influence_per_game / 50)
        elif element_type in [3, 4]:  # Midfielder/Forward
            # Need 12 CBIRT (harder threshold)
            threshold_prob = min(0.8, influence_per_game / 70)
        else:
            return 0  # GKs don't get CBIT bonus
        
        # Expected CBIT/CBIRT bonus
        expected_bonus = threshold_prob * self.POINTS_DEFENSIVE_BONUS
        
        # Adjust for fixture difficulty
        team_id = player['team']
        fixtures = self.fixture_map.get((team_id, gameweek), [])
        if fixtures:
            avg_difficulty = sum(f['difficulty'] for f in fixtures) / len(fixtures)
            # More defensive fixtures (high difficulty) = more defensive actions
            difficulty_multiplier = 0.8 + (avg_difficulty - 2) * 0.1
            expected_bonus *= max(0.7, min(1.3, difficulty_multiplier))
        
        return expected_bonus
    
    def calculate_appearance_points(self, player: pd.Series) -> float:
        """Estimate appearance points based on minutes history."""
        minutes = float(player.get('minutes', 0) or 0)
        games = max(1, int(player.get('starts', 0) or 0) + int(player.get('starts', 0) or 0) // 3)
        
        if minutes > 0 and games > 0:
            avg_mins = minutes / games
            if avg_mins >= 60:
                return 2.0
            elif avg_mins > 0:
                return 1.0
        return 0
    
    def calculate_regression_factor(self, player: pd.Series) -> float:
        """
        Calculate xG regression factor to identify under/over-performers.
        
        Returns:
            Factor > 1.0 = underperforming (likely to improve)
            Factor < 1.0 = overperforming (likely to regress)
            Factor = 1.0 = performing as expected
        """
        goals = float(player.get('goals_scored', 0) or 0)
        xg = float(player.get('expected_goals', 0) or 0)
        assists = float(player.get('assists', 0) or 0)
        xa = float(player.get('expected_assists', 0) or 0)
        
        # Need minimum sample for reliable regression
        if xg < 1.0 and xa < 0.5:
            return 1.0  # Not enough data
        
        # Calculate goal regression
        if xg > 0.5:
            goal_ratio = goals / xg
            # If ratio < 1, player has scored fewer than expected (underperformer)
            # If ratio > 1, player has scored more than expected (overperformer)
            goal_regression = 1.0 + (1.0 - goal_ratio) * 0.3  # Dampen effect
        else:
            goal_regression = 1.0
        
        # Calculate assist regression
        if xa > 0.3:
            assist_ratio = assists / xa
            assist_regression = 1.0 + (1.0 - assist_ratio) * 0.2
        else:
            assist_regression = 1.0
        
        # Combined regression factor (weighted)
        regression = (goal_regression * 0.7 + assist_regression * 0.3)
        
        # Clamp to reasonable range
        return max(0.7, min(1.4, regression))
    
    def calculate_bonus_points(self, player: pd.Series, gameweek: int) -> float:
        """
        Improved BPS bonus points prediction.
        
        Uses:
        - Position-specific bonus rates
        - ICT index correlation
        - Goals/assists boost
        """
        element_type = player['element_type']
        
        # Position-specific average bonus rates per GW
        # Based on historical data: GKs get less, mids/fwds who score get more
        BASE_BONUS_RATE = {
            1: 0.15,  # GKs rarely get bonus
            2: 0.35,  # Defenders get bonus for CS + tackles
            3: 0.50,  # Mids get bonus for goals/assists
            4: 0.60   # Fwds get bonus for goals
        }
        
        base_rate = BASE_BONUS_RATE.get(element_type, 0.4)
        
        # ICT index boost (high ICT = likely to get bonus)
        ict_index = float(player.get('ict_index', 0) or 0)
        minutes = max(float(player.get('minutes', 0) or 0), 1)
        
        # Calculate ICT per game
        games_played = minutes / 90
        if games_played > 0:
            ict_per_game = ict_index / games_played
            # Top players have ICT ~15+ per game
            ict_boost = min(1.0, ict_per_game / 20) * 0.5
        else:
            ict_boost = 0
        
        # Goals/assists boost
        goals = float(player.get('goals_scored', 0) or 0)
        assists = float(player.get('assists', 0) or 0)
        if games_played > 0:
            goal_rate = (goals + assists * 0.7) / games_played
            # Scoring players get bonus more often
            scoring_boost = min(0.8, goal_rate * 0.5)
        else:
            scoring_boost = 0
        
        # Combined expected bonus
        expected_bonus = base_rate + ict_boost + scoring_boost
        
        # Scale to 0-3 range (max 3 bonus points possible)
        return min(1.2, expected_bonus)  # Cap at ~1.2 expected per game
    
    def predict_gameweek(self, player: pd.Series, gameweek: int) -> Tuple[float, Dict[str, float]]:
        """
        Predict total expected points for a player in a specific gameweek.
        
        Returns:
            Tuple of (total_points, breakdown_dict)
        """
        # Calculate regression factor for xG/xA adjustments
        regression_factor = self.calculate_regression_factor(player)
        
        breakdown = {
            'xg_points': self.calculate_xg_points(player, gameweek) * regression_factor,
            'xa_points': self.calculate_xa_points(player, gameweek) * regression_factor,
            'cs_points': self.calculate_clean_sheet_prob(player, gameweek),
            'cbit_bonus': self.calculate_cbit_bonus(player, gameweek),
            'appearance': self.calculate_appearance_points(player),
            'bonus': self.calculate_bonus_points(player, gameweek),
            'regression_factor': regression_factor,
        }
        
        total = sum(v for k, v in breakdown.items() if k != 'regression_factor')
        
        # Dynamic calibration factor - decreases as season progresses
        # GW1: 0.85, GW19: 0.65, GW38: 0.50
        # This compensates for xG/xA accumulation making predictions drift upward
        dynamic_calibration = max(0.50, 0.85 - (gameweek - 1) * 0.01)
        
        total *= dynamic_calibration
        breakdown['calibration'] = dynamic_calibration
        
        # Form factor: Weight recent performance (last 4 games average)
        # 5 pts/game = baseline (factor = 1.0), higher = smaller boost, lower = bigger penalty
        # More aggressive to capture late-season form drops
        form = float(player.get('form', 0) or 0)
        if form > 0:
            form_factor = min(1.3, max(0.3, form / 5.0))  # 0.3 to 1.3 range
        else:
            form_factor = 0.5  # New/inactive players get strong penalty
        total *= form_factor
        breakdown['form_factor'] = form_factor
        
        # Minutes decay: Players losing playing time get lower predictions
        # Check recent 4 games - if averaging < 70 mins, apply penalty
        recent_4_mins = float(player.get('recent_4_minutes', 360) or 360)  # Default 4*90
        avg_recent_mins = recent_4_mins / 4.0
        if avg_recent_mins < 70:
            minutes_decay = max(0.3, avg_recent_mins / 70.0)
            total *= minutes_decay
            breakdown['minutes_decay'] = minutes_decay
        
        # Cap total prediction at realistic max (top player ~8 pts/gw on average)
        total = min(total, 8.0)
        
        # Adjust for availability (Injury news)
        # chance_of_playing_next_round is 0-100 or None (assumed 100)
        chance = player.get('chance_of_playing_next_round')
        if chance is not None and not pd.isna(chance):
            availability_factor = float(chance) / 100.0
            total *= availability_factor
            breakdown['availability_adj'] = availability_factor
            
        return total, breakdown
    
    def predict_multi_gameweek(self, player_id: int, gameweeks: int = 5) -> PointPrediction:
        """
        Generate multi-gameweek point predictions for a player.
        
        Args:
            player_id: Player's FPL element ID
            gameweeks: Number of gameweeks to predict
            
        Returns:
            PointPrediction with per-GW breakdown
        """
        player = self.players[self.players['id'] == player_id].iloc[0]
        
        gw_predictions = {}
        all_breakdowns = {}
        
        for gw in range(self.current_gw, self.current_gw + gameweeks):
            total, breakdown = self.predict_gameweek(player, gw)
            gw_predictions[gw] = total
            all_breakdowns[gw] = breakdown
        
        # Aggregate breakdown
        agg_breakdown = {}
        for key in ['xg_points', 'xa_points', 'cs_points', 'cbit_bonus', 'appearance', 'bonus']:
            agg_breakdown[key] = sum(b.get(key, 0) for b in all_breakdowns.values())
        
        return PointPrediction(
            player_id=player_id,
            player_name=player['web_name'],
            position=player.get('position', 'UNK'),
            team=self.team_names.get(player['team'], 'Unknown'),
            gameweek_predictions=gw_predictions,
            total_expected=sum(gw_predictions.values()),
            breakdown=agg_breakdown
        )
    
    def predict_all_players(self, gameweeks: int = 5) -> pd.DataFrame:
        """
        Generate predictions for all players.
        
        Returns:
            DataFrame with expected points per gameweek for all players
        """
        predictions = []
        
        for _, player in self.players.iterrows():
            player_id = player['id']
            
            # Calculate per-GW predictions
            gw_cols = {}
            total_xp = 0
            
            for gw in range(self.current_gw, self.current_gw + gameweeks):
                xp, _ = self.predict_gameweek(player, gw)
                gw_cols[f'xp_gw{gw}'] = xp
                total_xp += xp
            
            predictions.append({
                'id': player_id,
                'web_name': player['web_name'],
                'team': player['team'],
                'team_name': self.team_names.get(player['team'], 'Unknown'),
                'element_type': player['element_type'],
                'position': player.get('position', 'UNK'),
                'now_cost': player['now_cost'],
                'total_xp': total_xp,
                **gw_cols
            })
        
        return pd.DataFrame(predictions)


def compute_uncertainty_bounds(predictions_df: pd.DataFrame, 
                                confidence_level: float = 0.14) -> pd.DataFrame:
    """
    Compute uncertainty bounds for robust optimization.
    
    Given the R² ≈ 0.14 in FPL point predictions, we compute
    standard deviation estimates for each prediction.
    
    Args:
        predictions_df: DataFrame with point predictions
        confidence_level: R² value representing prediction accuracy
        
    Returns:
        DataFrame with additional columns for lower/upper bounds
    """
    df = predictions_df.copy()
    
    # Standard deviation estimate based on R²
    # If R² = 0.14, then about 86% of variance is unexplained
    # Estimate σ as proportional to expected points
    variance_ratio = 1 - confidence_level  # 0.86
    
    for col in df.columns:
        if col.startswith('xp_gw') or col == 'total_xp':
            # σ ≈ mean * sqrt(unexplained_variance)
            sigma = df[col] * np.sqrt(variance_ratio)
            df[f'{col}_lower'] = df[col] - 1.5 * sigma
            df[f'{col}_upper'] = df[col] + 1.5 * sigma
            df[f'{col}_sigma'] = sigma
    
    return df
