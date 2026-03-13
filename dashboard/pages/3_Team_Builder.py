#!/usr/bin/env python3
"""
Team Builder Page with FPL API Integration

Loads real FPL team data and provides optimization recommendations.
"""

import sys
import json
import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'backend'))

# FPL API Client
class FPLAPIClient:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_bootstrap(self):
        """Get static data."""
        resp = self.session.get(f"{self.BASE_URL}/bootstrap-static/", timeout=30)
        return resp.json()
    
    def get_manager_team(self, manager_id, gameweek=None):
        """Get manager's team for a specific gameweek."""
        if gameweek is None:
            # Get current gameweek
            bootstrap = self.get_bootstrap()
            current_event = next((e for e in bootstrap['events'] if e.get('is_current')), None)
            if current_event:
                gameweek = current_event['id']
            else:
                gameweek = 1
        
        url = f"{self.BASE_URL}/entry/{manager_id}/event/{gameweek}/picks/"
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return None
    
    def get_manager_history(self, manager_id):
        """Get manager's history."""
        url = f"{self.BASE_URL}/entry/{manager_id}/history/"
        resp = self.session.get(url, timeout=30)
        return resp.json() if resp.status_code == 200 else None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_fpl_data():
    """Load FPL static data."""
    client = FPLAPIClient()
    return client.get_bootstrap()


def load_exp032_model():
    """Load EXP-032 model."""
    model_path = Path(__file__).parent.parent.parent / 'models/exp032_fdr/model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def predict_player_points(player, model_data, fpl_data):
    """Predict points for a player using EXP-032."""
    if model_data is None:
        return 3.0  # Default prediction
    
    # Extract features
    try:
        # Get player form
        form = float(player.get('form', 0))
        
        # Get value
        value = player.get('now_cost', 50) / 10
        
        # Get transfer data
        transfers_in = player.get('transfers_in_event', 0)
        transfers_out = player.get('transfers_out_event', 0)
        transfers_balance = (transfers_in - transfers_out) / 1000
        
        # Get ownership
        selected = player.get('selected_by_percent', 0)
        log_selected = np.log1p(float(selected) * 1000)
        
        # Get position
        position_code = player.get('element_type', 3)
        
        # Build feature vector (simplified for now)
        features = np.array([[
            form,  # form_3gw proxy
            form,  # form_5gw proxy
            value,
            1,  # was_home (default)
            log_selected,
            transfers_balance,
            1 if position_code == 1 else 0,  # GK
            1 if position_code == 2 else 0,  # DEF
            1 if position_code == 3 else 0,  # MID
            1 if position_code == 4 else 0,  # FWD
            0.5,  # gameweek_norm
            3,  # fdr (default medium)
            1.0,  # opp_attack
            1.0,  # opp_defense
            1.0,  # own_attack
            1.0,  # own_defense
            0.0,  # strength_diff
            3.0   # rel_difficulty
        ]])
        
        # Scale and predict
        scaler = model_data['scaler']
        model = model_data['model']
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return max(0, prediction)  # Don't predict negative points
    except Exception as e:
        return 3.0  # Fallback


def render_team_builder():
    """Render Team Builder page."""
    st.markdown('<p class="main-header">🏃 Team Builder</p>', unsafe_allow_html=True)
    
    # Load FPL data
    with st.spinner("Loading FPL data..."):
        fpl_data = load_fpl_data()
    
    # Load model
    model_data = load_exp032_model()
    
    # Team ID Input
    st.subheader("🔑 Load Your FPL Team")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        team_id = st.text_input(
            "FPL Team ID",
            value=st.session_state.get('team_id', '9777842'),
            help="Find your Team ID in the FPL URL: fantasy.premierleague.com/entry/[ID]/"
        )
    
    with col2:
        gameweek = st.number_input("Gameweek", min_value=1, max_value=38, value=30)
    
    if st.button("Load Team", type="primary"):
        with st.spinner("Fetching team from FPL API..."):
            client = FPLAPIClient()
            team_data = client.get_manager_team(team_id, gameweek)
            
            if team_data:
                st.session_state['team_data'] = team_data
                st.session_state['team_id'] = team_id
                st.success(f"✅ Team loaded successfully!")
            else:
                st.error("❌ Failed to load team. Check your Team ID and try again.")
    
    # Display team if loaded
    if 'team_data' in st.session_state:
        team_data = st.session_state['team_data']
        
        st.divider()
        st.subheader("📋 Your Current Team")
        
        # Create player dataframe
        players_list = []
        total_predicted = 0
        
        for pick in team_data.get('picks', []):
            player_id = pick['element']
            
            # Find player in FPL data
            player = next((p for p in fpl_data['elements'] if p['id'] == player_id), None)
            
            if player:
                # Predict points
                predicted = predict_player_points(player, model_data, fpl_data)
                total_predicted += predicted
                
                # Get position name
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                position = position_map.get(player['element_type'], 'UNK')
                
                players_list.append({
                    'Position': position,
                    'Player': f"{player['first_name']} {player['second_name']}",
                    'Team': player.get('team', ''),
                    'Price': f"£{player['now_cost']/10:.1f}m",
                    'Form': player.get('form', '0'),
                    'Predicted': f"{predicted:.1f}",
                    'Captain': '⭐' if pick['is_captain'] else '',
                    ' Vice': '⭐' if pick['is_vice_captain'] else ''
                })
        
        if players_list:
            df = pd.DataFrame(players_list)
            st.dataframe(df, use_container_width=True)
            
            # Summary stats
            st.metric("Total Predicted Points", f"{total_predicted:.1f}")
        
        st.divider()
        
        # Transfer Recommendations
        st.subheader("💡 Transfer Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Players to Consider Selling:**")
            # Find low-performing players in team
            for pick in team_data.get('picks', [])[:3]:
                player_id = pick['element']
                player = next((p for p in fpl_data['elements'] if p['id'] == player_id), None)
                if player:
                    form = float(player.get('form', 0))
                    if form < 2.0:
                        st.warning(f"⚠️ {player['first_name']} {player['second_name']} (Form: {form})")
        
        with col2:
            st.markdown("**Top Transfer Targets:**")
            # Find high-form players not in team
            team_player_ids = [p['element'] for p in team_data.get('picks', [])]
            
            top_picks = []
            for player in fpl_data['elements']:
                if player['id'] not in team_player_ids:
                    form = float(player.get('form', 0))
                    if form > 6.0:
                        predicted = predict_player_points(player, model_data, fpl_data)
                        top_picks.append((predicted, player))
            
            top_picks.sort(reverse=True)
            for _, player in top_picks[:3]:
                st.success(f"✅ {player['first_name']} {player['second_name']} (Form: {player.get('form', 0)})")
    
    else:
        st.info("👆 Enter your FPL Team ID and click 'Load Team' to see your squad.")
        
        # Example
        with st.expander("How to find your Team ID"):
            st.markdown("""
            1. Go to [fantasy.premierleague.com](https://fantasy.premierleague.com)
            2. Log in and click 'Points'
            3. Look at the URL: `fantasy.premierleague.com/entry/9777842/...`
            4. Your Team ID is the number after `/entry/` (e.g., 9777842)
            """)


# Run the page
render_team_builder()
