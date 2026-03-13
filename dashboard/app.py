#!/usr/bin/env python3
"""
FPL Dashboard - Main Application

Streamlit-based dashboard for FPL model monitoring and team optimization.

Usage:
    cd dashboard && streamlit run app.py

Pages:
    - Overview: High-level metrics and status
    - Model Lab: Compare models and run predictions
    - Team Builder: Build optimal teams
    - Research: Track experiments and agents
    - Analytics: Deep dive into data
    - Settings: Configuration
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="FPL Optimizer Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .champion-badge {
        background-color: #ffd700;
        color: #000;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_model_metrics():
    """Load current model metrics."""
    metrics_path = Path(__file__).parent.parent / 'models/exp031_clean/metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_experiment_results():
    """Load recent experiment results."""
    results_dir = Path(__file__).parent.parent / 'research/agents'
    results = []
    
    # Agent 1 results
    agent1_dir = results_dir / 'agent1_results'
    if agent1_dir.exists():
        for f in agent1_dir.glob('production_test_*.json'):
            with open(f) as fp:
                results.append(json.load(fp))
    
    # Agent 3 & 4 results
    for agent_id in [3, 4]:
        report_path = results_dir / f'agent{agent_id}_results/final_report.json'
        if report_path.exists():
            with open(report_path) as fp:
                results.append(json.load(fp))
    
    return results


def render_overview():
    """Render Overview page."""
    st.markdown('<p class="main-header">🏠 FPL Optimizer Dashboard</p>', unsafe_allow_html=True)
    
    # Champion Model Card
    st.subheader("🏆 Current Champion")
    
    metrics = load_model_metrics()
    if metrics and 'ridge' in metrics:
        ridge = metrics['ridge']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Model",
                value="EXP-031"
            )
        
        with col2:
            st.metric(
                label="Spearman Correlation",
                value=f"{ridge['spearman']:.4f}",
                delta="+279% vs EXP-030"
            )
        
        with col3:
            st.metric(
                label="RMSE",
                value=f"{ridge['test_rmse']:.4f}"
            )
        
        with col4:
            st.metric(
                label="Training Samples",
                value="52,974"
            )
    else:
        st.warning("Model metrics not found. Please train EXP-031 first.")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Experiments",
            value="40+"
        )
    
    with col2:
        st.metric(
            label="Active Agents",
            value="4"
        )
    
    with col3:
        st.metric(
            label="Data Seasons",
            value="4 (2020-2024)"
        )
    
    with col4:
        st.metric(
            label="Production Improvement",
            value="91.4%"
        )
    
    st.divider()
    
    # Recent Activity
    st.subheader("📝 Recent Activity")
    
    results = load_experiment_results()
    if results:
        for result in results[-5:]:  # Show last 5
            if 'timestamp' in result:
                ts = result['timestamp']
                if 'experiment' in result:
                    st.info(f"📅 {ts[:10]} - {result.get('experiment', 'Unknown')}")
                elif 'agent_id' in result:
                    st.info(f"🤖 Agent {result['agent_id']}: {result.get('total_experiments', 0)} experiments completed")
    else:
        st.info("No recent activity. Run parallel agents to see results here.")
    
    st.divider()
    
    # Status Summary
    st.subheader("🎯 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.success("✅ EXP-031 Deployed")
    
    with status_col2:
        st.info("🔄 Ralph Loop Active")
    
    with status_col3:
        st.warning("⏳ 2024-25 Data Collection Pending")


def render_model_lab():
    """Render Model Lab page."""
    st.markdown('<p class="main-header">🔬 Model Lab</p>', unsafe_allow_html=True)
    
    # Model Comparison
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['EXP-030', 'EXP-031', 'EXP-031 (RF)', 'EXP-031 (GB)'],
        'Spearman': [0.1915, 0.7263, 0.7349, 0.7308],
        'RMSE': [0.8284, 1.4629, 1.4633, 1.4820],
        'Samples': [233, 52974, 52974, 52974],
        'Status': ['Legacy', '🏆 Champion', 'Alternative', 'Alternative']
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Feature Importance
    st.subheader("Feature Importance (EXP-031)")
    
    feature_data = {
        'Feature': ['form_3gw', 'transfers_balance', 'log_selected', 'position', 'value', 'was_home'],
        'Importance': [0.78, 0.06, 0.06, 0.05, 0.03, 0.02]
    }
    
    fig = px.bar(
        feature_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top Features by Importance',
        color='Importance',
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Prediction Playground
    st.subheader("🔮 Prediction Playground")
    
    st.info("Enter player details to get predicted points (simplified demo)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        form_3gw = st.slider("3-Game Form", 0.0, 15.0, 5.0, 0.5)
        value = st.slider("Player Value (£m)", 3.5, 15.0, 7.0, 0.1)
    
    with col2:
        was_home = st.checkbox("Home Fixture", value=True)
        position = st.selectbox("Position", ['GK', 'DEF', 'MID', 'FWD'])
    
    if st.button("Predict Points"):
        # Simple formula based on feature importance
        base = form_3gw * 0.78
        home_bonus = 0.5 if was_home else 0
        position_factor = {'GK': 0.5, 'DEF': 0.8, 'MID': 1.0, 'FWD': 1.2}[position]
        
        prediction = (base + home_bonus) * position_factor
        
        st.success(f"Predicted Points: {prediction:.2f}")
        st.caption("Note: This is a simplified demo. Use production model for real predictions.")


def render_team_builder():
    """Render Team Builder page."""
    st.markdown('<p class="main-header">🏃 Team Builder</p>', unsafe_allow_html=True)
    
    st.info("🚧 This feature requires FPL API integration. Coming in Phase 2.")
    
    # Placeholder for team selection
    st.subheader("Your Team")
    
    team_id = st.text_input("FPL Team ID", value="9777842")
    
    if st.button("Load Team"):
        st.warning("Team loading not yet implemented. Requires FPL API integration.")
    
    st.divider()
    
    # Optimization Settings
    st.subheader("Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.slider("Available Transfers", 0, 5, 1)
    
    with col2:
        st.slider("Bank Balance (£m)", 0.0, 5.0, 1.0, 0.1)
    
    with col3:
        st.selectbox("Formation", ['3-4-3', '3-5-2', '4-4-2', '4-3-3', '5-4-1', '5-3-2'])
    
    if st.button("Optimize Team"):
        st.info("Team optimization requires production model integration.")


def render_research():
    """Render Research page."""
    st.markdown('<p class="main-header">🔍 Research</p>', unsafe_allow_html=True)
    
    # Agent Status
    st.subheader("Agent Status")
    
    agent_data = {
        'Agent': ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'],
        'Task': [
            'Production Test',
            'Data Collection',
            'Ralph Loop (Features)',
            'Ralph Loop (Deep Learning)'
        ],
        'Status': ['✅ Completed', '✅ Completed', '✅ Completed', '✅ Completed'],
        'Results': [
            '91.4% improvement',
            'Using FPL API now',
            '20 experiments',
            '20 experiments'
        ]
    }
    
    st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
    
    st.divider()
    
    # Experiment Summary
    st.subheader("Experiment Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Experiments", "40")
    
    with col2:
        st.metric("Improvements Found", "0")
    
    with col3:
        st.metric("New Champions", "0")
    
    st.info("📊 No improvements over EXP-031 baseline found yet. Next: Implement real FDR features.")
    
    st.divider()
    
    # Hypothesis Queue
    st.subheader("🧪 Hypothesis Queue")
    
    hypotheses = [
        ("H1: FDR Features", "Add fixture difficulty ratings", "High"),
        ("H2: Team Strength", "Add team attack/defense ratings", "High"),
        ("H3: LSTM Model", "Neural network with sequences", "Medium"),
        ("H4: Position Models", "Separate models per position", "Medium"),
    ]
    
    for name, desc, priority in hypotheses:
        with st.expander(f"{name} (Priority: {priority})"):
            st.write(desc)
            st.button(f"Run {name}", key=name)


def render_analytics():
    """Render Analytics page."""
    st.markdown('<p class="main-header">📊 Analytics</p>', unsafe_allow_html=True)
    
    # Dataset Stats
    st.subheader("Dataset Statistics")
    
    stats_data = {
        'Metric': [
            'Total Samples',
            'Unique Players',
            'Seasons',
            'Avg Points/GW',
            'Max Points (Single GW)'
        ],
        'Value': [
            '52,974',
            '1,619',
            '4 (2020-2024)',
            '1.07',
            '24'
        ]
    }
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    st.divider()
    
    # Position Distribution
    st.subheader("Position Distribution")
    
    position_data = {
        'Position': ['GK', 'DEF', 'MID', 'FWD'],
        'Count': [5969, 17524, 22175, 7306],
        'Percentage': [11.3, 33.1, 41.9, 13.8]
    }
    
    fig = px.pie(
        position_data,
        values='Count',
        names='Position',
        title='Samples by Position',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)


def render_settings():
    """Render Settings page."""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    
    # Team Configuration
    st.subheader("Team Configuration")
    
    team_id = st.text_input(
        "FPL Team ID",
        value=st.session_state.get('team_id', '9777842'),
        help="Your FPL team ID from the URL"
    )
    
    if st.button("Save Team ID"):
        st.session_state['team_id'] = team_id
        st.success(f"Team ID saved: {team_id}")
    
    st.divider()
    
    # Refresh Settings
    st.subheader("Auto-Refresh Settings")
    
    refresh_rate = st.selectbox(
        "Refresh Rate",
        ['Manual', 'Every 15 minutes', 'Every 30 minutes', 'Every hour'],
        index=2
    )
    
    st.session_state['refresh_rate'] = refresh_rate
    
    st.divider()
    
    # Theme
    st.subheader("Appearance")
    
    theme = st.radio(
        "Theme",
        ['Light', 'Dark'],
        index=0
    )
    
    st.divider()
    
    # About
    st.subheader("About")
    
    st.markdown("""
    **FPL Lineup Optimizer Dashboard**
    
    Version: 1.0.0  
    Champion Model: EXP-031  
    Last Updated: 2026-03-13
    
    Built with:
    - Streamlit
    - Plotly
    - Scikit-learn
    - Pandas
    
    [GitHub Repository](https://github.com/AkshitPareek/fpl-lineup-optimizer)
    """)


def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.title("⚽ FPL Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ['🏠 Overview', '🔬 Model Lab', '🏃 Team Builder', 
         '🔍 Research', '📊 Analytics', '⚙️ Settings']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Render selected page
    if page == '🏠 Overview':
        render_overview()
    elif page == '🔬 Model Lab':
        render_model_lab()
    elif page == '🏃 Team Builder':
        render_team_builder()
    elif page == '🔍 Research':
        render_research()
    elif page == '📊 Analytics':
        render_analytics()
    elif page == '⚙️ Settings':
        render_settings()


if __name__ == '__main__':
    main()
