# Data Collection Strategy: Multi-League & Transfer Handling

> **Critical Question:** How do we handle teams and players that move between leagues?

---

## 🎯 The Challenge

### Problem 1: League Dynamics
```
2023/24: Luton Town (Championship) → Premier League
2024/25: Luton Town (Premier League) → Championship (relegated)

Q: Is Luton's "strength" comparable across leagues?
A: No - Championship Luton ≠ Premier League Luton
```

### Problem 2: Player Mobility
```
2023/24: Cole Palmer at Manchester City (PL)
2024/25: Cole Palmer at Chelsea (PL)

Q: How do we track his performance across teams?
Q: What about his form at new team vs old team?
```

### Problem 3: New Players
```
2024/25: New signing from La Liga (no PL history)
2024/25: Promoted from Championship (limited PL history)

Q: How do we predict for players with no PL data?
```

---

## 💡 Solutions

### 1. Multi-League Data Collection

**Collect from multiple leagues:**
- Premier League (primary target)
- Championship (for promoted teams)
- La Liga, Serie A, Bundesliga (for transfers in)
- Europa League, Champions League (European experience)

**Benefits:**
- Larger dataset (10,000+ samples vs 200)
- Learn general football patterns
- Handle league transitions

**Challenges:**
- Different scoring systems
- Different competition levels
- Data normalization required

### 2. League-Normalized Features

```python
# Instead of raw points, use percentiles within league
player['points_percentile_pl'] = player['total_points'] / pl_average * 100
player['points_percentile_championship'] = player['total_points'] / champ_average * 100

# League strength adjustment
player['adjusted_points'] = player['points'] * league_strength_factor
```

### 3. Transfer-Aware Features

```python
class TransferAwareFeatures:
    def calculate_transfer_features(self, player_id, current_team, current_gw):
        """
        Features that handle player transfers.
        """
        # 1. Same-team history (most reliable)
        same_team_history = self.get_performance_at_team(
            player_id, 
            current_team,
            n_games=10
        )
        
        # 2. Previous team history (adaptation factor)
        previous_team_history = self.get_performance_at_previous_teams(
            player_id,
            current_season=False
        )
        
        # 3. Time at current team (adaptation period)
        weeks_at_team = self.get_weeks_at_team(player_id, current_team, current_gw)
        
        # 4. Transfer recency
        games_since_transfer = self.get_games_since_transfer(player_id, current_gw)
        
        # 5. Position change indicator
        position_change = self.has_position_changed(player_id, current_team)
        
        return {
            'same_team_form': np.mean(same_team_history) if same_team_history else 0,
            'previous_team_form': np.mean(previous_team_history) if previous_team_history else 0,
            'weeks_at_team': weeks_at_team,
            'games_since_transfer': games_since_transfer,
            'transfer_recency': 1.0 / (games_since_transfer + 1),
            'position_change': 1.0 if position_change else 0.0,
            'new_team_adjustment': self.calculate_new_team_factor(weeks_at_team)
        }
    
    def calculate_new_team_factor(self, weeks_at_team):
        """
        Players often take 4-6 weeks to adapt to new team.
        """
        if weeks_at_team < 4:
            return 0.7  # Still adapting
        elif weeks_at_team < 8:
            return 0.85  # Getting settled
        else:
            return 1.0  # Fully adapted
```

### 4. Team-Level Features (League-Agnostic)

```python
class TeamFeatures:
    def calculate_team_features(self, team_id, league='PL'):
        """
        Features that work across leagues.
        """
        # Relative strength within league
        team_strength = self.get_league_relative_strength(team_id, league)
        
        # Recent form (last 5 games in ANY competition)
        recent_form = self.get_cross_competition_form(team_id, n_games=5)
        
        # Squad stability (fewer transfers = more predictable)
        squad_changes = self.get_squad_changes(team_id, window='1 year')
        
        # Manager stability
        manager_tenure = self.get_manager_tenure(team_id)
        
        # Home/away performance (league-agnostic)
        home_advantage = self.calculate_home_advantage(team_id)
        
        return {
            'relative_strength': team_strength,
            'recent_form_cross_comp': recent_form,
            'squad_stability': 1.0 / (squad_changes + 1),
            'manager_stability': min(1.0, manager_tenure / 52),  # weeks
            'home_advantage': home_advantage
        }
```

### 5. Handling Promoted/Relegated Teams

```python
class LeagueTransitionHandler:
    def handle_promoted_team(self, team_id, from_league='Championship', to_league='PL'):
        """
        Estimate team strength after promotion.
        """
        # 1. Get performance in lower league
        lower_league_performance = self.get_team_performance(
            team_id, 
            league=from_league,
            season='last'
        )
        
        # 2. Apply promotion adjustment factor
        # Historically, promoted teams struggle initially
        historical_promoted_teams = self.get_historical_promoted_teams()
        avg_performance_drop = self.calculate_performance_drop(historical_promoted_teams)
        
        # 3. Squad quality check
        squad_quality = self.assess_squad_quality(team_id)
        
        # 4. Estimated PL-equivalent strength
        estimated_strength = (
            lower_league_performance * 0.7 +  # Historical drop
            squad_quality * 0.3               # Squad mitigates drop
        )
        
        return {
            'estimated_pl_strength': estimated_strength,
            'promotion_adjustment': 0.7,  # Expect 30% drop initially
            'adaptive_factor': 1.0,  # Increases as season progresses
            'is_newly_promoted': 1.0,
            'games_in_pl': 0  # Increases each week
        }
    
    def update_adaptive_factor(self, team_id, games_played_in_new_league):
        """
        As promoted team plays more games, we use actual PL data.
        """
        if games_played_in_new_league < 5:
            # Use estimated strength
            return {'adaptive_factor': 0.0, 'use_estimated': True}
        elif games_played_in_new_league < 15:
            # Blend estimated and actual
            blend = games_played_in_new_league / 15
            return {'adaptive_factor': blend, 'use_estimated': False}
        else:
            # Use actual PL data only
            return {'adaptive_factor': 1.0, 'use_estimated': False}
```

### 6. Cold Start Problem (New Players)

```python
class ColdStartHandler:
    def handle_new_player(self, player_id, source_league=None):
        """
        Handle players with no PL history.
        """
        if source_league and source_league != 'PL':
            # 1. Get performance in source league
            source_performance = self.get_player_performance(
                player_id, 
                league=source_league
            )
            
            # 2. Apply league adjustment
            league_quality_factor = self.get_league_quality_factor(source_league, 'PL')
            adjusted_performance = source_performance * league_quality_factor
            
            # 3. Transfer history (some players adapt better)
            similar_transfers = self.find_similar_transfers(
                source_league=source_league,
                position=self.get_position(player_id),
                age=self.get_age(player_id)
            )
            avg_adaptation = np.mean([t['adaptation_factor'] for t in similar_transfers])
            
            return {
                'source_league_performance': source_performance,
                'pl_adjusted_estimate': adjusted_performance,
                'expected_adaptation': avg_adaptation,
                'confidence': 0.3  # Low confidence for new players
            }
        else:
            # Complete unknown (youth player, obscure league)
            # Use very conservative estimate
            return {
                'pl_adjusted_estimate': self.get_position_average(self.get_position(player_id)),
                'expected_adaptation': 0.5,
                'confidence': 0.1  # Very low confidence
            }
```

---

## 📊 Proposed Data Architecture

```
datasets/
├── multi_league/
│   ├── premier_league/          # 2020-2025
│   ├── championship/            # 2020-2025  
│   ├── la_liga/                 # 2020-2025
│   ├── serie_a/                 # 2020-2025
│   └── bundesliga/              # 2020-2025
│
├── normalized/
│   ├── league_adjusted/         # Points normalized by league
│   ├── percentile_rankings/     # Within-league percentiles
│   └── cross_league_features/   # League-agnostic features
│
├── transfers/
│   ├── transfer_history.csv     # All player transfers
│   ├── adaptation_metrics/      # How players adapt
│   └── transfer_windows/        # When transfers happen
│
└── teams/
    ├── league_transitions/      # Promo/relegation history
    ├── squad_composition/       # Team rosters over time
    └── manager_history/         # Manager changes
```

---

## 🎓 Key Insights

### What Makes Cross-League Prediction Hard?

1. **Different Skill Levels**
   - Championship 20-goal striker ≠ PL 20-goal striker
   - Solution: Use percentiles, not raw numbers

2. **Different Play Styles**
   - La Liga: Technical, possession-based
   - PL: Physical, high-intensity
   - Solution: Style-adjusted features

3. **Sample Size**
   - Need 2-3 seasons for reliable patterns
   - Solution: Bayesian priors from similar players

### What Data Should We Prioritize?

**High Value:**
- ✅ PL historical data (2020-2025)
- ✅ Championship data (for promoted teams)
- ✅ Transfer history with performance
- ✅ Team strength metrics

**Medium Value:**
- 🟡 European league data (La Liga, Serie A)
- 🟡 Champions League performance
- 🟡 International performance

**Low Value (for now):**
- ⚪ Lower leagues (League 1, etc.)
- ⚪ Youth/U21 data
- ⚪ Pre-season friendly data

---

## 🚀 Implementation Plan

### Phase 1: Multi-Year PL Data (Week 1-2)
```python
# Collect 5 seasons of PL data
target: 5 seasons × 20 teams × 25 players = 2,500 player-seasons
expected samples: ~15,000 gameweek-player records
```

### Phase 2: Championship Integration (Week 3)
```python
# Add Championship data for promoted teams
# Focus on teams that got promoted
# Use to improve promoted team predictions
target: +2,000 samples from Championship
```

### Phase 3: Transfer Tracking (Week 4)
```python
# Build transfer database
# Track player moves between teams
# Calculate adaptation factors
target: 500+ tracked transfers
```

### Phase 4: Cross-League Validation (Week 5-6)
```python
# Test on players who moved leagues
# Validate league adjustment factors
# Fine-tune normalization
```

---

## ✅ Immediate Actions

1. **Create Data Pipeline**
```bash
python create_multi_league_dataset.py \
    --leagues=PL,Championship \
    --seasons=2020-2025 \
    --output=datasets/multi_league_v1/
```

2. **Build Transfer Database**
```bash
python build_transfer_db.py \
    --source=transfermarkt \
    --years=2020-2025 \
    --output=data/transfers/
```

3. **Calculate League Adjustment Factors**
```python
# Run analysis to determine:
# - Championship → PL conversion factor
# - La Liga → PL conversion factor
# - How long players take to adapt
```

---

## 🎯 Success Metrics

**Dataset Size:**
- Current: 192 training samples
- Target Phase 1: 2,000+ samples (PL only, multi-year)
- Target Phase 2: 5,000+ samples (PL + Championship)
- Target Phase 3: 10,000+ samples (Multiple leagues)

**Model Improvement:**
- EXP-030 baseline: RMSE 0.8284
- With more data: Target RMSE 0.75-0.80
- With features: Target RMSE 0.70-0.75

---

**Bottom Line:**
Yes, we need multi-league data AND smart handling of league transitions, player transfers, and promoted teams. The feature engineering module is ready - we just need more data to make it work.

**Recommendation:**
1. Start with 5 seasons of PL data (easiest)
2. Add Championship data for promoted teams
3. Build transfer tracking
4. Retry EXP-031 with 2,000+ samples
