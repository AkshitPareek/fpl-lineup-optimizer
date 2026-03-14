
# BETTING ODDS FEATURE INTEGRATION
# Add these features to your model training

# Odds data structure
def parse_odds_features(odds_data):
    """
    Convert betting odds to features.
    
    Common markets:
    - Match winner (1X2)
    - Over/Under 2.5 goals
    - Both teams to score
    - First goalscorer
    - Correct score
    """
    features = {
        # Implied probabilities (1/odds)
        'home_win_prob': 1 / odds_data.get('home_win', 2.0),
        'draw_prob': 1 / odds_data.get('draw', 3.5),
        'away_win_prob': 1 / odds_data.get('away_win', 3.0),
        
        # Market expectations
        'expected_goals': odds_data.get('expected_goals', 2.5),
        'over_2_5_prob': 1 / odds_data.get('over_2_5', 2.0),
        'btts_prob': 1 / odds_data.get('btts_yes', 2.0),
        
        # Team strength indicators
        'favorite_margin': abs(odds_data.get('home_win', 2.0) - odds_data.get('away_win', 3.0)),
        'market_confidence': 1 / odds_data.get('home_win', 2.0) + 1 / odds_data.get('away_win', 3.0),
    }
    return features

# Odds movement (wisdom of crowds)
def calculate_odds_movement(opening_odds, closing_odds):
    """
    Track how odds change over time.
    
    Sharp money detection:
    - If odds shorten (decrease), money is coming in
    - Large movements indicate insider knowledge
    """
    movement = {}
    for outcome in ['home_win', 'draw', 'away_win']:
        if outcome in opening_odds and outcome in closing_odds:
            open_odd = opening_odds[outcome]
            close_odd = closing_odds[outcome]
            
            # Negative = odds shortened (money coming in)
            movement[f'{outcome}_movement'] = close_odd - open_odd
            
            # Percentage change
            movement[f'{outcome}_movement_pct'] = (close_odd - open_odd) / open_odd
    
    return movement

# Player-specific odds
def parse_player_odds(player_name, odds_data):
    """
    Extract player-specific markets.
    
    Markets:
    - Anytime goalscorer
    - First goalscorer
    - Shot on target
    - Carded
    """
    player_odds = odds_data.get('players', {}).get(player_name, {})
    
    return {
        'anytime_scorer_prob': 1 / player_odds.get('anytime', 5.0),
        'first_scorer_prob': 1 / player_odds.get('first', 15.0),
        'shot_on_target_prob': 1 / player_odds.get('shot_target', 2.0),
        'card_prob': 1 / player_odds.get('carded', 4.0),
    }
