
# Betting Odds Integration Research

## Research Summary

### Hypothesis
Betting odds reflect collective market wisdom including:
- Expert analysis
- Insider information
- Statistical models
- Public sentiment

These signals should improve FPL predictions.

### Why Betting Odds Work

1. **Efficient Market Hypothesis**
   - Odds adjust to all available information
   - Billions of dollars traded globally
   - Bookmakers employ expert analysts

2. **Wisdom of Crowds**
   - Collective judgment often beats experts
   - Thousands of bettors contribute to pricing
   - Real money at stake reduces noise

3. **Information Asymmetry**
   - Insiders bet on non-public info
   - Injury news, team news leaks
   - Odds movement reveals this

### Data Sources

#### 1. Odds APIs (Paid)
- **Odds API** (odds-api.com)
  - $29/month for historical data
  - Real-time odds updates
  - Multiple bookmakers

- **Betfair API**
  - Exchange odds (most accurate)
  - Historical data available
  - Requires developer account

#### 2. Web Scraping (Free but risky)
- Oddschecker.com
- Bet365
- Pinnacle

**Note:** Check terms of service before scraping.

#### 3. Academic Datasets
- Football-data.co.uk
- Contains some historical odds
- Free but limited bookmakers

### Expected Features

#### Match-Level
- Home win probability
- Expected goals (from over/under markets)
- Market confidence (sum of implied probs)
- Odds movement indicators

#### Player-Level
- Anytime goalscorer probability
- First goalscorer odds
- Card probability
- Shot on target odds

### Expected Impact

| Feature Type | Expected Improvement |
|--------------|---------------------|
| Match odds | +0.5-1.0% Spearman |
| Odds movement | +0.3-0.5% Spearman |
| Player odds | +0.5-1.0% Spearman |
| **Total** | **+1.0-2.0% Spearman** |

### Implementation Plan

#### Phase 1: Data Collection (1 week)
- [ ] Sign up for Odds API
- [ ] Collect historical odds for 2024-25
- [ ] Match odds to FPL fixtures
- [ ] Store in database

#### Phase 2: Feature Engineering (3 days)
- [ ] Convert odds to implied probabilities
- [ ] Calculate odds movements
- [ ] Create player-specific features
- [ ] Test correlation with FPL points

#### Phase 3: Model Integration (3 days)
- [ ] Add odds features to training pipeline
- [ ] Retrain EXP-032
- [ ] Evaluate improvement
- [ ] Deploy if positive

### Alternative Approach

Instead of live odds, use:
- Opening odds (less influenced by public)
- Closing odds (most informed)
- Pinnacle odds (sharp bookmaker)
- Exchange odds (no bookmaker margin)

### Risks

1. **Overfitting**
   - Odds already incorporate public sentiment
   - May not add new information
   - Cross-validation essential

2. **Data Quality**
   - Different bookmakers have different margins
   - Line shopping affects odds
   - Need consistent data source

3. **Cost**
   - Historical odds data is expensive
   - $29-99/month for APIs
   - May not justify improvement

### Budget Option

Use closing line value (CLV) research:
- Academic papers show CLV predicts outcomes
- Can calculate from football-data.co.uk
- Free but less granular
