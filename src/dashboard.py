import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from real_time_navigation import RealTimeNavigationSystem
import time
import os
import subprocess
from datetime import datetime
import json

st.set_page_config(layout="wide", page_title="Harare RL Agent Dashboard")

# Initialize session state
if 'nav_system' not in st.session_state:
    st.session_state.nav_system = RealTimeNavigationSystem()
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False
if 'episode_count' not in st.session_state:
    st.session_state.episode_count = 0

st.title("🚦 Harare RL Agent - Real-Time Navigation Dashboard")
st.markdown("Advanced analytics and real-time visualization for PPO-trained navigation agent")

# Sidebar controls
with st.sidebar:
    st.header("🎮 Control Panel")
    
    # Run single episode
    if st.button("🚀 Run Single Episode", type="primary"):
        with st.spinner("Running episode..."):
            episode_data, episode_metrics = st.session_state.nav_system.run_episode_with_analytics()
            st.session_state.episode_count += 1
            st.success(f"Episode {st.session_state.episode_count} completed!")
            st.rerun()
    
    # Auto-run toggle
    auto_run = st.checkbox("🔄 Auto-run episodes", value=st.session_state.auto_run)
    st.session_state.auto_run = auto_run
    
    if auto_run:
        auto_interval = st.slider("Interval (seconds)", 1, 30, 5)
        if st.button("⏹️ Stop Auto-run"):
            st.session_state.auto_run = False
            st.rerun()
    
    st.divider()
    
    # Session controls
    if st.button("🔄 Reset Session"):
        st.session_state.nav_system = RealTimeNavigationSystem()
        st.session_state.episode_count = 0
        st.success("Session reset!")
        st.rerun()
    
    # TensorBoard integration
    st.header("📊 TensorBoard")
    if st.button("🚀 Launch TensorBoard"):
        try:
            # Start TensorBoard in background
            subprocess.Popen(["tensorboard", "--logdir", "logs/", "--port", "6006", "--host", "0.0.0.0"])
            st.success("TensorBoard launched! Access at: http://localhost:6006")
        except Exception as e:
            st.error(f"Failed to launch TensorBoard: {e}")
    
    st.markdown("[Open TensorBoard](http://localhost:6006)")

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    # Real-time navigation map
    st.subheader("🗺️ Real-Time Navigation")
    
    if st.session_state.nav_system.session_analytics['episode_history']:
        # Get latest episode data
        latest_episode_data = st.session_state.nav_system.session_analytics['episode_history'][-1]
        
        # Create and display animated map
        if hasattr(st.session_state.nav_system, 'current_episode_data'):
            animated_map = st.session_state.nav_system.create_animated_map(
                st.session_state.nav_system.current_episode_data, 
                latest_episode_data
            )
            # Save and display map
            animated_map.save("real_time_navigation.html")
            with open("real_time_navigation.html", "r") as f:
                map_html = f.read()
            html(map_html, height=500)
        else:
            # Fallback to static map
            if os.path.exists("harare_agent_path.html"):
                with open("harare_agent_path.html", "r") as f:
                    map_html = f.read()
                html(map_html, height=500)
            else:
                st.info("Run an episode to see navigation visualization")
    else:
        st.info("🎯 Click 'Run Single Episode' to start navigation visualization")

with col2:
    # Live metrics
    st.subheader("📈 Live Metrics")
    
    if st.session_state.nav_system.session_analytics['episode_history']:
        summary = st.session_state.nav_system.get_session_summary()
        
        # Key metrics
        st.metric("Total Episodes", summary['total_episodes'])
        st.metric("Success Rate", f"{summary['success_rate']:.1%}")
        st.metric("Avg Reward", f"{summary['avg_reward']:.2f}")
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
        
        # Latest episode info
        latest = summary['best_episode']
        st.divider()
        st.subheader("🏆 Best Episode")
        st.write(f"**Episode {latest['episode_id']}**")
        st.write(f"Reward: {latest['total_reward']:.2f}")
        st.write(f"Steps: {latest['total_steps']}")
        st.write(f"Confidence: {latest['avg_confidence']:.2f}")
    else:
        st.info("Run episodes to see live metrics")

# Analytics section
st.divider()
st.subheader("📊 Advanced Analytics")

if st.session_state.nav_system.session_analytics['episode_history']:
    # Generate charts
    fig_performance, fig_steps, fig_success, fig_exploration = st.session_state.nav_system.generate_analytics_charts()
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📏 Step Distribution", "🎯 Success Rate", "🔍 Exploration"])
    
    with tab1:
        if fig_performance:
            st.plotly_chart(fig_performance, use_container_width=True)
    
    with tab2:
        if fig_steps:
            st.plotly_chart(fig_steps, use_container_width=True)
    
    with tab3:
        if fig_success:
            st.plotly_chart(fig_success, use_container_width=True)
    
    with tab4:
        if fig_exploration:
            st.plotly_chart(fig_exploration, use_container_width=True)
    
    # Detailed episode table
    st.subheader("📋 Episode Details")
    df = pd.DataFrame(st.session_state.nav_system.session_analytics['episode_history'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('episode_id', ascending=False)
    
    # Format for display
    display_df = df[['episode_id', 'total_steps', 'total_reward', 'avg_confidence', 'success', 'exploration_score']].copy()
    display_df.columns = ['Episode', 'Steps', 'Reward', 'Confidence', 'Success', 'Exploration']
    display_df['Confidence'] = display_df['Confidence'].round(3)
    display_df['Reward'] = display_df['Reward'].round(2)
    display_df['Exploration'] = display_df['Exploration'].round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Export data
    if st.button("💾 Export Session Data"):
        export_data = {
            'session_summary': st.session_state.nav_system.get_session_summary(),
            'episode_history': st.session_state.nav_system.session_analytics['episode_history']
        }
        
        filename = f"harare_rl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        st.success(f"Session data exported to {filename}")
else:
    st.info("🎯 Run episodes to see detailed analytics")

# Auto-run functionality
if st.session_state.auto_run and 'auto_interval' in locals():
    time.sleep(auto_interval)
    with st.spinner("Running auto episode..."):
        episode_data, episode_metrics = st.session_state.nav_system.run_episode_with_analytics()
        st.session_state.episode_count += 1
    st.rerun()

# Footer
st.divider()
st.markdown("---")
st.markdown("🤖 **Harare RL Agent Dashboard** - Real-time navigation analytics with PPO reinforcement learning")