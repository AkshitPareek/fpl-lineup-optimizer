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

# FPL API Client with error handling
@st.cache_resource
def get_fpl_client():
    """Get FPL API client with session."""
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
    except Exception as e:
        st.error(f"Failed to initialize API client: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_fpl_bootstrap():
    """Load FPL static data with error handling."""
    try:
        import requests
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        resp = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API returned status {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to load FPL data: {e}")
        return None

@st.cache_data(ttl=300)
def load_manager_team(manager_id, gameweek=None):
    """Load manager team with error handling."""
    try:
        import requests
        
        # Get current gameweek if not specified
        if gameweek is None:
            bootstrap = load_fpl_bootstrap()
            if bootstrap:
                current_event = next((e for e in bootstrap.get('events', []) if e.get('is_current')), None)
                if current_event:
                    gameweek = current_event['id']
                else:
                    gameweek = 1
        
        url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gameweek}/picks/"
        resp = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            st.error(f"Team not found. Make sure Team ID {manager_id} exists and is public.")
            return None
        else:
            st.error(f"API Error: Status {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def load_sample_data():
    """Load sample team data for demonstration."""
    return {
        "picks": [
            {"element": 1, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "is_captain": False, "is_vice_captain": True},
            {"element": 3, "is_captain": False, "is_vice_captain": False},
            {"element": 4, "is_captain": False, "is_vice_captain": False},
            {"element": 5, "is_captain": False, "is_vice_captain": False},
            {"element": 6, "is_captain": False, "is_vice_captain": False},
            {"element": 7, "is_captain": False, "is_vice_captain": False},
            {"element": 8, "is_captain": False, "is_vice_captain": False},
            {"element": 9, "is_captain": False, "is_vice_captain": False},
            {"element": 10, "is_captain": False, "is_vice_captain": False},
            {"element": 11, "is_captain": False, "is_vice_captain": False},
        ],
        "entry_history": {"total_points": 1500}
    }


def render_team_builder():
    """Render Team Builder page."""
    st.markdown('<p class="main-header">🏃 Team Builder</p>', unsafe_allow_html=True)
    
    # Load FPL data
    with st.spinner("Loading FPL data..."):
        fpl_data = load_fpl_bootstrap()
    
    if fpl_data is None:
        st.error("❌ Failed to load FPL data. API may be temporarily unavailable.")
        st.info("Try again in a few minutes, or use the Sample Team below.")
        use_sample = True
    else:
        st.success("✅ FPL data loaded successfully!")
        use_sample = False
    
    # Team ID Input
    st.subheader("🔑 Load Your FPL Team")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        team_id = st.text_input(
            "FPL Team ID",
            value="9777842",
            help="Find your Team ID in the FPL URL: fantasy.premierleague.com/entry/[ID]/"
        )
    
    with col2:
        gameweek = st.number_input("Gameweek", min_value=1, max_value=38, value=29, help="Current: GW29")
    
    # Demo mode checkbox
    use_demo = st.checkbox("Use Demo Team (if API fails)", value=use_sample)
    
    if st.button("Load Team", type="primary"):
        if use_demo:
            st.session_state['team_data'] = load_sample_data()
            st.session_state['demo_mode'] = True
            st.success("✅ Demo team loaded!")
        else:
            with st.spinner("Fetching from FPL API..."):
                team_data = load_manager_team(team_id, gameweek)
                
                if team_data:
                    st.session_state['team_data'] = team_data
                    st.session_state['team_id'] = team_id
                    st.session_state['demo_mode'] = False
                    st.success(f"✅ Team loaded successfully!")
                else:
                    st.error("❌ Failed to load team.")
                    st.info("💡 Try enabling 'Use Demo Team' checkbox to see how it works.")
    
    # Display team if loaded
    if 'team_data' in st.session_state:
        team_data = st.session_state['team_data']
        demo_mode = st.session_state.get('demo_mode', False)
        
        if demo_mode:
            st.info("📢 Showing DEMO team. Enable 'Use Demo Team' and click Load to try with real data.")
        
        st.divider()
        st.subheader("📋 Your Team")
        
        players_list = []
        
        for pick in team_data.get('picks', []):
            player_id = pick['element']
            
            if fpl_data:
                player = next((p for p in fpl_data.get('elements', []) if p['id'] == player_id), None)
            else:
                player = None
            
            if player:
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                position = position_map.get(player.get('element_type', 3), 'UNK')
                
                players_list.append({
                    'Position': position,
                    'Player': f"{player.get('first_name', '')} {player.get('second_name', '')}",
                    'Price': f"£{player.get('now_cost', 0)/10:.1f}m",
                    'Form': player.get('form', '0'),
                    'Captain': '⭐' if pick.get('is_captain') else ''
                })
            else:
                # Demo fallback
                players_list.append({
                    'Position': 'MID',
                    'Player': f'Player {player_id}',
                    'Price': '£7.0m',
                    'Form': '5.0',
                    'Captain': '⭐' if pick.get('is_captain') else ''
                })
        
        if players_list:
            df = pd.DataFrame(players_list)
            st.dataframe(df, use_container_width=True)
        
        # Transfer Recommendations
        st.divider()
        st.subheader("💡 Transfer Recommendations")
        
        if fpl_data:
            st.markdown("**Top Form Players to Consider:**")
            
            # Find high-form players
            top_picks = []
            for player in fpl_data.get('elements', [])[:50]:  # Check first 50
                form = float(player.get('form', 0))
                if form > 7.0:
                    top_picks.append({
                        'Player': f"{player.get('first_name', '')} {player.get('second_name', '')}",
                        'Team': player.get('team', ''),
                        'Form': form,
                        'Price': f"£{player.get('now_cost', 0)/10:.1f}m"
                    })
            
            if top_picks:
                top_df = pd.DataFrame(top_picks[:5])
                st.dataframe(top_df, use_container_width=True)
            else:
                st.info("No high-form players found in sample.")
        else:
            st.info("FPL data not available for recommendations.")
    
    else:
        st.info("👆 Enter your FPL Team ID and click 'Load Team'")
        
        with st.expander("How to find your Team ID"):
            st.markdown("""
            1. Go to [fantasy.premierleague.com](https://fantasy.premierleague.com)
            2. Log in and click 'Points'
            3. Look at the URL: `fantasy.premierleague.com/entry/9777842/...`
            4. Your Team ID is the number after `/entry/` (e.g., **9777842**)
            
            **Note:** Your team must be set to public to load via API.
            """)


# Run the page
render_team_builder()
