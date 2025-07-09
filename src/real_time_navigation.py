import folium
import streamlit as st
import numpy as np
import pandas as pd
import time
import json
from harare_env import HarareEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import threading
import queue

class RealTimeNavigationSystem:
    def __init__(self, model_path="ppo_harare_rl_agent"):
        self.env = HarareEnv(
            edge_file="../data/harare_graph.gpkg",
            node_file="../data/harare_graph.gpkg",
            feature_file="../data/tile_features.pkl"
        )
        try:
            self.model = PPO.load(model_path)
            if self.model is None:
                raise ValueError(f"Failed to load model from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model from {model_path}: {str(e)}")
        self.current_episode_data = []
        self.session_analytics = {
            'total_episodes': 0,
            'total_steps': 0,
            'avg_reward': 0,
            'success_rate': 0,
            'episode_history': []
        }
        
    def run_episode_with_analytics(self, max_steps=50):
        """Run a single episode and collect detailed analytics"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            
        episode_data = {
            'steps': [],
            'rewards': [],
            'actions': [],
            'edges': [],
            'positions': [],
            'timestamps': [],
            'confidence_scores': []
        }
        
        # Store episode data for map creation
        self.current_episode_data = episode_data
        
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < max_steps:
            start_time = time.time()
            
            # Get action and confidence from model
            if self.model is None:
                raise ValueError("Model is not loaded")
            
            action, _states = self.model.predict(obs.reshape(1, -1), deterministic=False)
            action = int(action[0])
            
            # Get action probabilities for confidence score
            try:
                obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
                self.model.policy.set_training_mode(False)
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.detach().cpu().numpy()
                confidence = float(np.max(action_probs))
            except Exception as e:
                # Fallback to simple confidence measure
                confidence = 0.5
            
            # Execute action
            step_result = self.env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Get current position
            try:
                u, v, key = self.env.current_edge
                row = self.env.edges.loc[(u, v, key)]
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    position = coords[0]  # Start of edge
                else:
                    position = None
            except:
                position = None
            
            # Store step data
            episode_data['steps'].append(step_count)
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action)
            episode_data['edges'].append(self.env.current_edge)
            episode_data['positions'].append(position)
            episode_data['timestamps'].append(time.time() - start_time)
            episode_data['confidence_scores'].append(confidence)
            
            total_reward += reward
            step_count += 1
            
        # Calculate episode metrics
        episode_metrics = {
            'episode_id': len(self.session_analytics['episode_history']),
            'total_steps': step_count,
            'total_reward': total_reward,
            'avg_reward': total_reward / step_count if step_count > 0 else 0,
            'success': done and reward > 0,
            'avg_confidence': np.mean(episode_data['confidence_scores']),
            'exploration_score': len(set(episode_data['edges'])) / len(episode_data['edges']) if episode_data['edges'] else 0,
            'completion_time': sum(episode_data['timestamps']),
            'timestamp': datetime.now()
        }
        
        # Update session analytics
        self.session_analytics['total_episodes'] += 1
        self.session_analytics['total_steps'] += step_count
        self.session_analytics['episode_history'].append(episode_metrics)
        
        # Calculate running averages
        if self.session_analytics['episode_history']:
            self.session_analytics['avg_reward'] = np.mean([ep['avg_reward'] for ep in self.session_analytics['episode_history']])
            self.session_analytics['success_rate'] = np.mean([ep['success'] for ep in self.session_analytics['episode_history']])
        
        return episode_data, episode_metrics
    
    def create_animated_map(self, episode_data, episode_metrics):
        """Create an animated map showing agent movement"""
        # Create base map
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        # Add path with different colors for different rewards
        for i, (edge, reward, confidence) in enumerate(zip(episode_data['edges'], episode_data['rewards'], episode_data['confidence_scores'])):
            try:
                u, v, key = edge
                row = self.env.edges.loc[(u, v, key)]
                
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    
                    # Color based on reward and confidence
                    if reward > 0:
                        color = 'green' if confidence > 0.7 else 'lightgreen'
                    elif reward == 0:
                        color = 'blue' if confidence > 0.7 else 'lightblue'
                    else:
                        color = 'red' if confidence > 0.7 else 'pink'
                    
                    # Width based on step order (thicker for later steps)
                    weight = 3 + (i / len(episode_data['edges'])) * 3
                    
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in coords],
                        color=color,
                        weight=weight,
                        opacity=0.8,
                        popup=f"Step {i+1}: Reward={reward:.2f}, Confidence={confidence:.2f}"
                    ).add_to(m)
                    
            except Exception as e:
                print(f"Error plotting edge {edge}: {e}")
        
        # Add start and end markers
        if episode_data['positions']:
            start_pos = episode_data['positions'][0]
            end_pos = episode_data['positions'][-1]
            
            if start_pos:
                folium.Marker(
                    [start_pos[1], start_pos[0]],
                    popup=f"Start - Episode {episode_metrics['episode_id']}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
            
            if end_pos and end_pos != start_pos:
                folium.Marker(
                    [end_pos[1], end_pos[0]],
                    popup=f"End - Total Reward: {episode_metrics['total_reward']:.2f}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
        
        return m
    
    def generate_analytics_charts(self):
        """Generate comprehensive analytics charts"""
        if not self.session_analytics['episode_history']:
            return None, None, None, None
        
        df = pd.DataFrame(self.session_analytics['episode_history'])
        
        # Episode performance over time
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(
            x=df['episode_id'],
            y=df['total_reward'],
            mode='lines+markers',
            name='Total Reward',
            line=dict(color='blue')
        ))
        fig_performance.add_trace(go.Scatter(
            x=df['episode_id'],
            y=df['avg_confidence'],
            mode='lines+markers',
            name='Avg Confidence',
            yaxis='y2',
            line=dict(color='green')
        ))
        fig_performance.update_layout(
            title='Agent Performance Over Time',
            xaxis_title='Episode',
            yaxis_title='Total Reward',
            yaxis2=dict(
                title='Average Confidence',
                overlaying='y',
                side='right'
            )
        )
        
        # Step distribution
        fig_steps = px.histogram(
            df, 
            x='total_steps', 
            title='Distribution of Episode Lengths',
            labels={'total_steps': 'Steps per Episode', 'count': 'Frequency'}
        )
        
        # Success rate over time
        df['success_rate_rolling'] = df['success'].rolling(window=5, min_periods=1).mean()
        fig_success = px.line(
            df, 
            x='episode_id', 
            y='success_rate_rolling',
            title='Success Rate (5-episode rolling average)',
            labels={'episode_id': 'Episode', 'success_rate_rolling': 'Success Rate'}
        )
        
        # Exploration vs Performance scatter
        fig_exploration = px.scatter(
            df,
            x='exploration_score',
            y='total_reward',
            size='total_steps',
            color='avg_confidence',
            title='Exploration vs Performance',
            labels={
                'exploration_score': 'Exploration Score (Unique Edges / Total Edges)',
                'total_reward': 'Total Reward',
                'avg_confidence': 'Average Confidence'
            }
        )
        
        return fig_performance, fig_steps, fig_success, fig_exploration
    
    def get_session_summary(self):
        """Get comprehensive session summary"""
        if not self.session_analytics['episode_history']:
            return {}
        
        df = pd.DataFrame(self.session_analytics['episode_history'])
        
        return {
            'total_episodes': self.session_analytics['total_episodes'],
            'total_steps': self.session_analytics['total_steps'],
            'avg_reward': self.session_analytics['avg_reward'],
            'success_rate': self.session_analytics['success_rate'],
            'best_episode': df.loc[df['total_reward'].idxmax()].to_dict(),
            'avg_steps_per_episode': df['total_steps'].mean(),
            'avg_confidence': df['avg_confidence'].mean(),
            'avg_exploration_score': df['exploration_score'].mean(),
            'total_session_time': df['completion_time'].sum()
        }