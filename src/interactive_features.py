import streamlit as st
import folium
import numpy as np
from streamlit_folium import st_folium
import random

class InteractiveRouteManager:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.start_edge = None
        self.goal_edge = None
        
    def create_interactive_map(self):
        """Create an interactive map for selecting start and goal points"""
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        # Add some sample edges as clickable markers
        sample_edges = random.sample(list(self.env.edges.index), min(50, len(self.env.edges)))
        
        for i, edge in enumerate(sample_edges):
            try:
                u, v, key = edge
                row = self.env.edges.loc[edge]
                
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    center = coords[len(coords)//2] if len(coords) > 1 else coords[0]
                    
                    folium.Marker(
                        [center[1], center[0]],  # lat, lon
                        popup=f"Edge: {u}-{v}-{key}",
                        tooltip=f"Click to select as start/goal",
                        icon=folium.Icon(color='blue', icon='circle')
                    ).add_to(m)
                    
            except Exception as e:
                continue
        
        return m
    
    def run_guided_navigation(self, start_edge, goal_edge, max_steps=100):
        """Run navigation from start to goal with guidance"""
        # Reset environment to start edge
        self.env.reset()
        self.env.current_edge = start_edge
        
        obs = self.env._get_observation()
        path = [start_edge]
        rewards = []
        actions = []
        confidence_scores = []
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Check if we reached the goal
            if self.env.current_edge == goal_edge:
                rewards.append(10)  # Goal reward
                done = True
                break
            
            # Get action from model
            action, _states = self.model.predict(obs.reshape(1, -1), deterministic=False)
            action = int(action[0])
            
            # Get confidence
            obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            with self.model.policy.set_training_mode(False):
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.cpu().numpy()
                confidence = float(np.max(action_probs))
            
            # Take step
            step_result = self.env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            path.append(self.env.current_edge)
            rewards.append(reward)
            actions.append(action)
            confidence_scores.append(confidence)
            
            step_count += 1
        
        return {
            'path': path,
            'rewards': rewards,
            'actions': actions,
            'confidence_scores': confidence_scores,
            'reached_goal': self.env.current_edge == goal_edge,
            'total_steps': step_count
        }
    
    def create_route_comparison_interface(self):
        """Create interface for comparing different routing strategies"""
        st.subheader("🛣️ Route Comparison Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strategy 1: RL Agent**")
            if st.button("Run RL Navigation"):
                # Run RL agent
                pass
        
        with col2:
            st.write("**Strategy 2: Random Walk**")
            if st.button("Run Random Navigation"):
                # Run random navigation
                pass
    
    def run_random_navigation(self, start_edge, max_steps=50):
        """Run random navigation for comparison"""
        self.env.reset()
        self.env.current_edge = start_edge
        
        path = [start_edge]
        rewards = []
        actions = []
        
        for step in range(max_steps):
            # Random action
            action = random.randint(0, self.env.action_space.n - 1)
            
            step_result = self.env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            path.append(self.env.current_edge)
            rewards.append(reward)
            actions.append(action)
            
            if done:
                break
        
        return {
            'path': path,
            'rewards': rewards,
            'actions': actions,
            'total_steps': len(path) - 1,
            'strategy': 'random'
        }
    
    def calculate_route_metrics(self, route_data):
        """Calculate comprehensive metrics for a route"""
        metrics = {
            'total_distance': len(route_data['path']) - 1,
            'total_reward': sum(route_data['rewards']) if 'rewards' in route_data else 0,
            'avg_reward': np.mean(route_data['rewards']) if 'rewards' in route_data else 0,
            'success_rate': 1 if route_data.get('reached_goal', False) else 0,
            'unique_edges': len(set(route_data['path'])),
            'exploration_ratio': len(set(route_data['path'])) / len(route_data['path']) if route_data['path'] else 0,
            'avg_confidence': np.mean(route_data['confidence_scores']) if 'confidence_scores' in route_data else 0
        }
        
        return metrics
    
    def create_route_analysis_dashboard(self, route_data_list):
        """Create comprehensive route analysis dashboard"""
        if not route_data_list:
            st.info("No route data available for analysis")
            return
        
        # Calculate metrics for all routes
        metrics_list = [self.calculate_route_metrics(route) for route in route_data_list]
        
        # Create comparison table
        import pandas as pd
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.index = [f"Route {i+1}" for i in range(len(df_metrics))]
        
        st.subheader("📊 Route Comparison Analysis")
        st.dataframe(df_metrics.round(3))
        
        # Create visualization
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Radar chart comparison
        if len(metrics_list) > 1:
            fig = go.Figure()
            
            metrics_to_compare = ['total_reward', 'exploration_ratio', 'avg_confidence', 'success_rate']
            
            for i, metrics in enumerate(metrics_list):
                values = [metrics.get(metric, 0) for metric in metrics_to_compare]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_to_compare,
                    fill='toself',
                    name=f'Route {i+1}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max([max([m.get(metric, 0) for metric in metrics_to_compare]) 
                                      for m in metrics_list])]
                    )),
                title="Route Performance Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        return df_metrics

class AdvancedAnalytics:
    def __init__(self):
        pass
    
    def create_performance_heatmap(self, session_data):
        """Create performance heatmap over time"""
        import plotly.graph_objects as go
        
        if not session_data['episode_history']:
            return None
        
        # Prepare data for heatmap
        episodes = []
        metrics = []
        values = []
        
        for episode in session_data['episode_history']:
            ep_id = episode['episode_id']
            
            for metric, value in episode.items():
                if isinstance(value, (int, float)) and metric != 'episode_id':
                    episodes.append(ep_id)
                    metrics.append(metric)
                    values.append(value)
        
        if not episodes:
            return None
        
        # Create heatmap
        import pandas as pd
        df = pd.DataFrame({
            'Episode': episodes,
            'Metric': metrics,
            'Value': values
        })
        
        pivot_df = df.pivot(index='Metric', columns='Episode', values='Value')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Performance Metrics Heatmap',
            xaxis_title='Episode',
            yaxis_title='Metric'
        )
        
        return fig
    
    def create_decision_analysis(self, episode_data):
        """Analyze agent decision patterns"""
        if not episode_data.get('actions'):
            return None
        
        import plotly.express as px
        import pandas as pd
        
        # Action distribution
        action_counts = pd.Series(episode_data['actions']).value_counts().sort_index()
        
        fig_actions = px.bar(
            x=action_counts.index,
            y=action_counts.values,
            title='Action Distribution',
            labels={'x': 'Action', 'y': 'Frequency'}
        )
        
        # Confidence vs Reward correlation
        if 'confidence_scores' in episode_data and 'rewards' in episode_data:
            fig_correlation = px.scatter(
                x=episode_data['confidence_scores'],
                y=episode_data['rewards'],
                title='Confidence vs Reward Correlation',
                labels={'x': 'Confidence Score', 'y': 'Reward'},
                trendline='ols'
            )
            
            return fig_actions, fig_correlation
        
        return fig_actions, None