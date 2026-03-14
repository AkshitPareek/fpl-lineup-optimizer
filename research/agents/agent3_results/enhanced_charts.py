
def render_points_timeline(team_data, fpl_data):
    """Show expected points timeline."""
    import plotly.graph_objects as go
    
    st.subheader("📈 Expected Points Timeline")
    
    # Simulate next 5 gameweeks
    gameweeks = list(range(30, 35))
    expected_points = [45, 52, 48, 55, 50]  # Simulated
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gameweeks,
        y=expected_points,
        mode='lines+markers',
        name='Expected Points',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.update_layout(
        title='Projected Points - Next 5 Gameweeks',
        xaxis_title='Gameweek',
        yaxis_title='Expected Points',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
