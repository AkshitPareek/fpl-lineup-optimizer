
def render_chip_advisor(team_data, fpl_data, gameweek):
    """Chip usage advisor component."""
    st.subheader("🎲 Chip Usage Advisor")
    
    chips = {
        'wildcard': {'name': 'Wildcard', 'icon': '🃏', 'recommendation': ''},
        'freehit': {'name': 'Free Hit', 'icon': '🎯', 'recommendation': ''},
        'bboost': {'name': 'Bench Boost', 'icon': '💺', 'recommendation': ''},
        '3xc': {'name': 'Triple Captain', 'icon': '👑', 'recommendation': ''}
    }
    
    # Simple recommendations based on gameweek
    if gameweek <= 19:
        chips['wildcard']['recommendation'] = "⏳ Consider saving for second half"
        chips['freehit']['recommendation'] = "⏳ Good for blank gameweeks"
    else:
        chips['wildcard']['recommendation'] = "✅ Good time to use (second half)"
    
    # Bench boost recommendation
    if team_data:
        bench_players = team_data.get('picks', [])[11:15]
        if len(bench_players) >= 3:
            chips['bboost']['recommendation'] = "✅ Good if bench is strong"
        else:
            chips['bboost']['recommendation'] = "⏳ Wait for double gameweek"
    
    # Triple captain
    chips['3xc']['recommendation'] = "⏳ Save for player with double gameweek"
    
    # Display
    for chip_id, chip_info in chips.items():
        with st.expander(f"{chip_info['icon']} {chip_info['name']}"):
            st.write(chip_info['recommendation'])
