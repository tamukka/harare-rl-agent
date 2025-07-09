import folium
import streamlit as st
import time
import json
from folium import plugins
import numpy as np
from datetime import datetime

class AnimatedNavigationVisualizer:
    def __init__(self, env):
        self.env = env
        self.base_map = None
        
    def create_live_navigation_map(self, episode_data, episode_metrics, animation_speed=1.0):
        """Create an animated map showing step-by-step navigation"""
        # Create base map
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        # Prepare animation data
        animation_data = []
        
        for i, (edge, reward, confidence, action) in enumerate(zip(
            episode_data['edges'], 
            episode_data['rewards'], 
            episode_data['confidence_scores'],
            episode_data['actions']
        )):
            try:
                u, v, key = edge
                row = self.env.edges.loc[(u, v, key)]
                
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    
                    # Color coding based on performance
                    if reward > 0:
                        color = '#2E8B57' if confidence > 0.7 else '#90EE90'  # Dark/light green
                        icon_color = 'green'
                    elif reward == 0:
                        color = '#1E90FF' if confidence > 0.7 else '#87CEEB'  # Dark/light blue
                        icon_color = 'blue'
                    else:
                        color = '#DC143C' if confidence > 0.7 else '#FFB6C1'  # Dark/light red
                        icon_color = 'red'
                    
                    # Create step marker
                    center_coord = coords[len(coords)//2] if len(coords) > 1 else coords[0]
                    
                    # Add step marker with detailed popup
                    popup_content = f"""
                    <b>Step {i+1}</b><br>
                    Reward: {reward:.2f}<br>
                    Confidence: {confidence:.3f}<br>
                    Action: {action}<br>
                    Edge: {u} → {v}
                    """
                    
                    folium.Marker(
                        [center_coord[1], center_coord[0]],
                        popup=folium.Popup(popup_content, max_width=200),
                        icon=folium.Icon(
                            color=icon_color, 
                            icon='circle',
                            prefix='fa'
                        )
                    ).add_to(m)
                    
                    # Add path segment
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in coords],
                        color=color,
                        weight=4 + (i / len(episode_data['edges'])) * 2,  # Increasing thickness
                        opacity=0.8,
                        popup=f"Step {i+1}: {reward:.2f} reward"
                    ).add_to(m)
                    
            except Exception as e:
                st.error(f"Error plotting edge {edge}: {e}")
        
        # Add episode summary
        episode_summary = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px;
                    background-color: white; 
                    border: 2px solid #ccc; 
                    border-radius: 5px; 
                    padding: 10px;
                    z-index: 1000;">
            <h4>Episode {episode_metrics['episode_id']}</h4>
            <p><b>Total Steps:</b> {episode_metrics['total_steps']}</p>
            <p><b>Total Reward:</b> {episode_metrics['total_reward']:.2f}</p>
            <p><b>Avg Confidence:</b> {episode_metrics['avg_confidence']:.3f}</p>
            <p><b>Success:</b> {'✅' if episode_metrics['success'] else '❌'}</p>
        </div>
        """
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:green"></i> Good Reward<br>
        <i class="fa fa-circle" style="color:blue"></i> Neutral<br>
        <i class="fa fa-circle" style="color:red"></i> Negative Reward<br>
        <br><b>Line Thickness:</b> Step Order
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_comparison_map(self, episode_data_list, episode_metrics_list):
        """Create a map comparing multiple episodes"""
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
                 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        for ep_idx, (episode_data, episode_metrics) in enumerate(zip(episode_data_list, episode_metrics_list)):
            color = colors[ep_idx % len(colors)]
            
            # Create feature group for this episode
            fg = folium.FeatureGroup(name=f"Episode {episode_metrics['episode_id']} (Reward: {episode_metrics['total_reward']:.1f})")
            
            for i, edge in enumerate(episode_data['edges']):
                try:
                    u, v, key = edge
                    row = self.env.edges.loc[(u, v, key)]
                    
                    if hasattr(row.geometry, 'coords'):
                        coords = list(row.geometry.coords)
                        
                        folium.PolyLine(
                            locations=[(lat, lon) for lon, lat in coords],
                            color=color,
                            weight=3,
                            opacity=0.7
                        ).add_to(fg)
                        
                except Exception as e:
                    continue
            
            fg.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_heatmap_visualization(self, session_analytics):
        """Create a heatmap of frequently visited edges"""
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        # Count edge visits across all episodes
        edge_counts = {}
        edge_positions = {}
        
        for episode in session_analytics['episode_history']:
            if 'edges' in episode:  # This might need adjustment based on data structure
                for edge in episode['edges']:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
                    
                    # Get edge position
                    try:
                        u, v, key = edge
                        row = self.env.edges.loc[(u, v, key)]
                        if hasattr(row.geometry, 'coords'):
                            coords = list(row.geometry.coords)
                            center = coords[len(coords)//2] if len(coords) > 1 else coords[0]
                            edge_positions[edge] = [center[1], center[0]]  # lat, lon
                    except:
                        continue
        
        # Create heatmap data
        heat_data = []
        max_count = max(edge_counts.values()) if edge_counts else 1
        
        for edge, count in edge_counts.items():
            if edge in edge_positions:
                # Normalize count and add to heatmap
                intensity = count / max_count
                heat_data.append([edge_positions[edge][0], edge_positions[edge][1], intensity])
        
        if heat_data:
            # Add heatmap
            plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        return m
    
    def create_performance_overlay_map(self, episode_data, episode_metrics):
        """Create a map with performance metrics overlay"""
        m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)
        
        # Add performance metrics as chart overlay
        rewards = episode_data['rewards']
        confidence_scores = episode_data['confidence_scores']
        
        # Create mini charts for overlay
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Reward chart
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            y=rewards,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig_reward.update_layout(
            title="Step Rewards",
            width=300,
            height=150,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        
        # Confidence chart
        fig_confidence = go.Figure()
        fig_confidence.add_trace(go.Scatter(
            y=confidence_scores,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        fig_confidence.update_layout(
            title="Confidence Scores",
            width=300,
            height=150,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        
        # Convert to HTML and embed
        reward_html = pio.to_html(fig_reward, include_plotlyjs='cdn', div_id="reward_chart")
        confidence_html = pio.to_html(fig_confidence, include_plotlyjs='cdn', div_id="confidence_chart")
        
        # Add path
        for i, edge in enumerate(episode_data['edges']):
            try:
                u, v, key = edge
                row = self.env.edges.loc[(u, v, key)]
                
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    
                    # Color based on reward
                    reward = episode_data['rewards'][i]
                    if reward > 0:
                        color = 'green'
                    elif reward == 0:
                        color = 'blue'
                    else:
                        color = 'red'
                    
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in coords],
                        color=color,
                        weight=4,
                        opacity=0.8
                    ).add_to(m)
                    
            except Exception as e:
                continue
        
        return m