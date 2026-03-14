
# Weather Impact on FPL Performance

## Research Summary

### Hypothesis
Weather conditions significantly impact match outcomes and player performance,
providing predictive signal for FPL points.

### Expected Impacts

#### Rain
- **Effect:** Slippery conditions, slower ball movement
- **GK/DEF:** More saves, harder to defend
- **MID/FWD:** Reduced passing accuracy
- **Expected impact:** -5% to -10% overall points

#### Wind (>20 km/h)
- **Effect:** Affects long shots, crosses, high balls
- **DEF:** Harder to clear, more errors
- **MID:** Crossing less effective
- **Expected impact:** -3% to -7% for wingers

#### Temperature
- **Hot (>25°C):** Fatigue, dehydration
  - More substitutions
  - Lower intensity in second half
  - Expected: -5% late in matches

- **Cold (<5°C):** Muscle stiffness
  - Slower reaction times
  - Higher injury risk
  - Expected: -3% overall

### Data Sources

1. **OpenWeatherMap API**
   - Free tier: 1000 calls/day
   - Historical data available
   - Stadium coordinates needed

2. **Visual Crossing Weather API**
   - F1 racing weather data
   - High accuracy
   - Paid tier required

3. **Historical Match Data**
   - Match reports with weather
   - Football-data.co.uk
   - Manual annotation

### Implementation Plan

#### Phase 1: Data Collection
- [ ] Get weather API key
- [ ] Map stadiums to locations
- [ ] Collect historical weather for past matches
- [ ] Merge with FPL dataset

#### Phase 2: Analysis
- [ ] Correlate weather with player performance
- [ ] Identify weather-sensitive players
- [ ] Build weather impact model

#### Phase 3: Integration
- [ ] Add weather features to EXP-032
- [ ] Retrain with weather data
- [ ] Evaluate improvement

### Expected Outcome
- **Target:** +1-2% Spearman improvement
- **Confidence:** Medium (requires data validation)
- **Timeline:** 1-2 weeks for full implementation

### Alternative: Weather as Post-Processing
Instead of training with weather, use it for:
- Captain pick adjustments
- Transfer timing
- Bench boost decisions
