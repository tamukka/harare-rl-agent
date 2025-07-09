import folium
import numpy as np
from harare_env import HarareEnv
from stable_baselines3 import PPO

# Load environment and model
env = HarareEnv(
    edge_file="../data/harare_graph.gpkg",
    node_file="../data/harare_graph.gpkg",
    feature_file="../data/tile_features.pkl"
)
model = PPO.load("ppo_harare_rl_agent")

# Reset environment and handle different return formats
reset_result = env.reset()
if isinstance(reset_result, tuple):
    # Gymnasium format: (observation, info)
    obs, info = reset_result
else:
    # Old Gym format: just observation
    obs = reset_result

# Initialize tracking variables
path = []  # Track the path taken
done = False
max_steps = 100  # Prevent infinite loops

print("Starting episode...")
step_count = 0

# Run one episode and record path
while not done and step_count < max_steps:
    # Add current edge to path
    path.append(env.current_edge)

    # Get action from model
    action, _ = model.predict(obs.reshape(1, -1))
    action = int(action[0])  # Convert numpy array to integer

    # Take step in environment
    step_result = env.step(action)

    # Handle different step return formats
    if len(step_result) == 4:
        # Old Gym format: (obs, reward, done, info)
        obs, reward, done, info = step_result
    elif len(step_result) == 5:
        # Gymnasium format: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated

    step_count += 1
    print(f"Step {step_count}: Edge {env.current_edge}, Reward: {reward:.2f}")

print(f"Episode finished after {step_count} steps")
print(f"Final path length: {len(path)}")

# Create map centered on Harare
m = folium.Map(location=[-17.8292, 31.0522], zoom_start=13)

# Plot path
plotted_edges = 0
for i, edge in enumerate(path):
    try:
        u, v, key = edge
        row = env.edges.loc[(u, v, key)]

        if hasattr(row.geometry, 'coords'):
            coords = list(row.geometry.coords)
            # Convert coordinates (lon, lat) to (lat, lon) for folium
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in coords],
                color='blue' if i < len(path)-1 else 'red',  # Last edge in red
                weight=4,
                opacity=0.7
            ).add_to(m)
            plotted_edges += 1
        else:
            print(f"No geometry for edge {edge}")

    except Exception as e:
        print(f"Failed to plot edge {edge}: {e}")

print(f"Successfully plotted {plotted_edges} out of {len(path)} edges")

# Add start and end markers
if path:
    try:
        # Start marker
        first_edge = path[0]
        u, v, key = first_edge
        row = env.edges.loc[(u, v, key)]
        if hasattr(row.geometry, 'coords'):
            coords = list(row.geometry.coords)
            start_lat, start_lon = coords[0][1], coords[0][0]
            folium.Marker(
                [start_lat, start_lon],
                popup="Start",
                icon=folium.Icon(color='green')
            ).add_to(m)

        # End marker
        last_edge = path[-1]
        u, v, key = last_edge
        row = env.edges.loc[(u, v, key)]
        if hasattr(row.geometry, 'coords'):
            coords = list(row.geometry.coords)
            end_lat, end_lon = coords[-1][1], coords[-1][0]
            folium.Marker(
                [end_lat, end_lon],
                popup="End",
                icon=folium.Icon(color='red')
            ).add_to(m)
    except Exception as e:
        print(f"Failed to add markers: {e}")

# Save map
m.save("harare_agent_path.html")
print("✅ Path saved to harare_agent_path.html")