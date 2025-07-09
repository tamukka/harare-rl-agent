import streamlit as st
import folium
from streamlit.components.v1 import html
import os

st.set_page_config(layout="wide", page_title="Harare RL Agent - Simple Dashboard")

st.title("🚦 Harare RL Agent Dashboard")
st.markdown("Real-time navigation visualization for PPO-trained agent")

# Simple test button
if st.button("🎯 Test Basic Navigation"):
    st.info("Running basic navigation test...")
    
    # Run the basic visualization
    import subprocess
    try:
        result = subprocess.run(["python", "visualize_path.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            st.success("✅ Navigation test completed!")
            st.text("Output:")
            st.code(result.stdout)
            
            # Display the generated map
            if os.path.exists("harare_agent_path.html"):
                st.subheader("🗺️ Agent Navigation Path")
                with open("harare_agent_path.html", "r") as f:
                    map_html = f.read()
                html(map_html, height=600)
            else:
                st.error("Map file not found")
        else:
            st.error(f"❌ Navigation test failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        st.error("❌ Test timed out")
    except Exception as e:
        st.error(f"❌ Error running test: {e}")

# TensorBoard section
st.divider()
st.subheader("📊 Training Monitoring")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Launch TensorBoard"):
        try:
            import subprocess
            subprocess.Popen(["tensorboard", "--logdir", "logs/", "--port", "6006"])
            st.success("TensorBoard started! Access at: http://localhost:6006")
        except Exception as e:
            st.error(f"Failed to launch TensorBoard: {e}")

with col2:
    st.markdown("[Open TensorBoard](http://localhost:6006)")

# Training status
st.subheader("🔄 Training Status")
log_dirs = ["logs/PPO_1", "logs/PPO_2", "logs/PPO_3", "logs/PPO_4"]
existing_logs = [d for d in log_dirs if os.path.exists(d)]

if existing_logs:
    st.success(f"✅ Found {len(existing_logs)} training runs")
    for log_dir in existing_logs:
        st.write(f"📁 {log_dir}")
else:
    st.warning("⚠️ No training logs found. Run training first!")

# Simple metrics display
st.subheader("📈 Quick Metrics")
if os.path.exists("training.log"):
    with open("training.log", "r") as f:
        lines = f.readlines()
        if lines:
            st.text("Last few training log lines:")
            st.code("".join(lines[-10:]))
else:
    st.info("No training log found")

st.divider()
st.markdown("🤖 **Simple Dashboard** - Basic navigation testing and monitoring")