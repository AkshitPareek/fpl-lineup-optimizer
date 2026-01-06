"""
Live Gameweek Tracker Module

Provides real-time gameweek analytics:
- Live points tracking
- Rank estimations
- Captain performance analysis
- Bonus point predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime


@dataclass
class LivePlayerStatus:
    """Live status of a player in the current gameweek."""
    player_id: int
    player_name: str
    team: str
    position: str
    live_points: int
    bonus_predicted: int
    minutes_played: int
    goals: int
    assists: int
    clean_sheet_potential: bool
    fixture_status: str  # 'playing', 'finished', 'upcoming'


class LiveTracker:
    """
    Tracks live gameweek performance.
    
    Features:
    - Live point totals
    - Estimated rank movements
    - Bonus point predictions
    - Top captain performance
    """
    
    def __init__(self, players_df: pd.DataFrame, teams_df: pd.DataFrame,
                 live_data: Dict = None):
        """
        Initialize with FPL data.
        
        Args:
            players_df: FPL players DataFrame
            teams_df: FPL teams DataFrame
            live_data: Live gameweek data (optional)
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        self.live_data = live_data or {}
        
        self.team_names = dict(zip(self.teams['id'], self.teams['name']))
        self.team_short_names = dict(zip(self.teams['id'], self.teams['short_name']))
        self.positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    def get_live_prices(self) -> pd.DataFrame:
        """
        Get current player prices and price change predictions.
        
        Returns:
            DataFrame with price info and change indicators
        """
        data = []
        
        for _, player in self.players.iterrows():
            # Calculate transfer activity indicator
            transfers_in = int(player.get('transfers_in_event', 0) or 0)
            transfers_out = int(player.get('transfers_out_event', 0) or 0)
            net_transfers = transfers_in - transfers_out
            
            # Cost change this week
            cost_change = float(player.get('cost_change_event', 0) or 0) / 10
            
            # Price prediction based on transfer activity
            # Rough threshold: ~100k net transfers for price change
            if net_transfers > 50000:
                price_trend = 'RISING'
                rise_probability = min(95, 50 + net_transfers / 10000 * 5)
            elif net_transfers < -50000:
                price_trend = 'FALLING'
                rise_probability = max(5, 50 + net_transfers / 10000 * 5)
            else:
                price_trend = 'STABLE'
                rise_probability = 50
            
            data.append({
                'id': player['id'],
                'web_name': player['web_name'],
                'team': self.team_short_names.get(player['team'], 'UNK'),
                'position': self.positions.get(player['element_type'], 'UNK'),
                'current_price': player['now_cost'] / 10,
                'cost_change_gw': cost_change,
                'transfers_in': transfers_in,
                'transfers_out': transfers_out,
                'net_transfers': net_transfers,
                'price_trend': price_trend,
                'ownership': float(player.get('selected_by_percent', 0) or 0)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('net_transfers', ascending=False)
    
    def get_price_risers(self, min_net_transfers: int = 20000) -> pd.DataFrame:
        """Get players likely to rise in price."""
        df = self.get_live_prices()
        risers = df[df['net_transfers'] >= min_net_transfers]
        return risers.sort_values('net_transfers', ascending=False)
    
    def get_price_fallers(self, min_net_out: int = 20000) -> pd.DataFrame:
        """Get players likely to fall in price."""
        df = self.get_live_prices()
        fallers = df[df['net_transfers'] <= -min_net_out]
        return fallers.sort_values('net_transfers')
    
    def get_top_transfers_in(self, limit: int = 20) -> pd.DataFrame:
        """Get most transferred in players this GW."""
        df = self.get_live_prices()
        return df.nlargest(limit, 'transfers_in')
    
    def get_top_transfers_out(self, limit: int = 20) -> pd.DataFrame:
        """Get most transferred out players this GW."""
        df = self.get_live_prices()
        return df.nlargest(limit, 'transfers_out')
    
    def get_form_analysis(self) -> pd.DataFrame:
        """
        Analyze current form versus historical averages.
        
        Identifies players in hot/cold streaks.
        """
        data = []
        
        for _, player in self.players.iterrows():
            form = float(player.get('form', 0) or 0)
            ppg = float(player.get('points_per_game', 0) or 0)
            
            # Form vs season average
            form_differential = form - ppg
            
            # Form trend
            if form_differential > 2:
                form_status = 'HOT STREAK'
            elif form_differential > 1:
                form_status = 'Good Form'
            elif form_differential < -2:
                form_status = 'COLD STREAK'
            elif form_differential < -1:
                form_status = 'Poor Form'
            else:
                form_status = 'Average'
            
            if form >= 3:  # Only include relevant players
                data.append({
                    'id': player['id'],
                    'web_name': player['web_name'],
                    'team': self.team_short_names.get(player['team'], 'UNK'),
                    'position': self.positions.get(player['element_type'], 'UNK'),
                    'price': player['now_cost'] / 10,
                    'form': form,
                    'points_per_game': ppg,
                    'form_differential': round(form_differential, 2),
                    'form_status': form_status,
                    'total_points': int(player.get('total_points', 0) or 0),
                    'ownership': float(player.get('selected_by_percent', 0) or 0)
                })
        
        df = pd.DataFrame(data)
        return df.sort_values('form', ascending=False)
    
    def get_hot_streaks(self, min_form: float = 6.0) -> pd.DataFrame:
        """Get players on hot streaks (high form)."""
        df = self.get_form_analysis()
        return df[df['form'] >= min_form]
    
    def get_breakout_candidates(self) -> pd.DataFrame:
        """
        Identify potential breakout candidates.
        
        Players with high recent form but low ownership.
        """
        df = self.get_form_analysis()
        
        breakouts = df[
            (df['form'] >= 5.0) &  # Good recent form
            (df['ownership'] < 10.0) &  # Low ownership
            (df['price'] <= 8.0)  # Affordable
        ]
        
        return breakouts.sort_values('form', ascending=False)
    
    def get_value_plays(self) -> pd.DataFrame:
        """
        Get high-value plays (form/price ratio).
        
        Players returning above their price point.
        """
        data = []
        
        for _, player in self.players.iterrows():
            form = float(player.get('form', 0) or 0)
            price = player['now_cost'] / 10
            
            if form >= 3.0 and price >= 4.0:
                # Value = form points per million
                value_ratio = form / price
                
                data.append({
                    'id': player['id'],
                    'web_name': player['web_name'],
                    'team': self.team_short_names.get(player['team'], 'UNK'),
                    'position': self.positions.get(player['element_type'], 'UNK'),
                    'price': price,
                    'form': form,
                    'value_ratio': round(value_ratio, 2),
                    'ownership': float(player.get('selected_by_percent', 0) or 0)
                })
        
        df = pd.DataFrame(data)
        return df.sort_values('value_ratio', ascending=False)
    
    def get_gameweek_summary(self, current_gw: int) -> Dict:
        """
        Get summary of gameweek activity.
        
        Args:
            current_gw: Current gameweek number
            
        Returns:
            Dict with GW summary
        """
        prices = self.get_live_prices()
        form = self.get_form_analysis()
        
        # Top performers this GW based on event points
        top_scorers = self.players.nlargest(10, 'event_points')[
            ['web_name', 'team', 'event_points']
        ].copy()
        top_scorers['team'] = top_scorers['team'].map(self.team_short_names)
        
        return {
            'gameweek': current_gw,
            'top_scorers': top_scorers.to_dict('records'),
            'most_transferred_in': prices.head(5)[
                ['web_name', 'team', 'transfers_in', 'net_transfers']
            ].to_dict('records'),
            'most_transferred_out': prices.tail(5).sort_values('net_transfers')[
                ['web_name', 'team', 'transfers_out', 'net_transfers']
            ].to_dict('records'),
            'hot_form': form[form['form_status'] == 'HOT STREAK'].head(5)[
                ['web_name', 'team', 'form', 'form_differential']
            ].to_dict('records'),
            'price_risers': prices[prices['price_trend'] == 'RISING'].head(5)[
                ['web_name', 'team', 'net_transfers', 'current_price']
            ].to_dict('records'),
            'price_fallers': prices[prices['price_trend'] == 'FALLING'].head(5)[
                ['web_name', 'team', 'net_transfers', 'current_price']
            ].to_dict('records')
        }
    
    def get_captain_analysis(self) -> pd.DataFrame:
        """
        Analyze captain options based on recent returns.
        
        Shows which captains have delivered.
        """
        data = []
        
        # Filter to viable captain options (mids/fwds with good ownership)
        for _, player in self.players.iterrows():
            element_type = player['element_type']
            if element_type not in [3, 4]:  # Only MID/FWD
                continue
            
            ownership = float(player.get('selected_by_percent', 0) or 0)
            if ownership < 5:  # Need reasonable ownership
                continue
            
            form = float(player.get('form', 0) or 0)
            event_points = int(player.get('event_points', 0) or 0)
            
            # Calculate captain value (2x points for captaincy)
            captain_value = event_points * 2
            
            data.append({
                'id': player['id'],
                'web_name': player['web_name'],
                'team': self.team_short_names.get(player['team'], 'UNK'),
                'price': player['now_cost'] / 10,
                'ownership': ownership,
                'form': form,
                'gw_points': event_points,
                'captain_value': captain_value,
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('captain_value', ascending=False)


def get_live_summary(players_df: pd.DataFrame, teams_df: pd.DataFrame,
                     current_gw: int) -> Dict:
    """Quick function for live summary."""
    tracker = LiveTracker(players_df, teams_df)
    return tracker.get_gameweek_summary(current_gw)
