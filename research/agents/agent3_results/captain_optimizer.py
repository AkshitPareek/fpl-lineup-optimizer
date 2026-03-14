
def render_captain_optimizer(team_data, fpl_data, model_data):
    """Captain pick optimizer component."""
    st.subheader("⭐ Captain Pick Optimizer")
    
    if not team_data or not fpl_data:
        st.info("Load your team first to see captain recommendations.")
        return
    
    candidates = []
    
    for pick in team_data.get('picks', [])[:11]:  # Only starting XI
        player_id = pick['element']
        player = next((p for p in fpl_data['elements'] if p['id'] == player_id), None)
        
        if player:
            # Predict points
            pred, conf = predict_player_points(player, model_data, fpl_data)
            
            # Calculate captaincy score (prediction * confidence)
            captain_score = pred * conf
            
            candidates.append({
                'player': f"{player['first_name']} {player['second_name']}",
                'position': player.get('element_type', 3),
                'predicted': pred,
                'confidence': conf,
                'captain_score': captain_score,
                'form': float(player.get('form', 0)),
                'fixture_difficulty': 3  # Simplified
            })
    
    # Sort by captain score
    candidates.sort(key=lambda x: x['captain_score'], reverse=True)
    
    # Display top 3
    st.markdown("**Top Captain Picks:**")
    
    for i, cand in enumerate(candidates[:3], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        st.success(f"{emoji} **{cand['player']}** - Predicted: {cand['predicted']:.1f} pts (Confidence: {cand['confidence']:.0%})")
    
    # Show all as table
    df = pd.DataFrame(candidates)
    st.dataframe(df[['player', 'predicted', 'confidence', 'captain_score']], use_container_width=True)
