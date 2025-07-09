import pickle
import numpy as np

# Load and inspect tile features
print("🔍 CHECKING TILE FEATURES")
print("=" * 40)

with open("../data/tile_features.pkl", "rb") as f:
    tile_features = pickle.load(f)

print(f"📊 Feature file loaded successfully")
print(f"   Number of tiles: {len(tile_features)}")
print(f"   Sample keys: {list(tile_features.keys())[:5]}")

# Check a few features
sample_keys = list(tile_features.keys())[:3]
for key in sample_keys:
    feature = tile_features[key]
    print(f"   {key}: shape={feature.shape}, range=[{feature.min():.3f}, {feature.max():.3f}]")

# Check if any features match current edge format
print(f"\n🎯 CHECKING EDGE FORMAT COMPATIBILITY")
from harare_env import HarareEnv

env = HarareEnv(
    edge_file="../data/harare_graph.gpkg",
    node_file="../data/harare_graph.gpkg", 
    feature_file="../data/tile_features.pkl"
)

# Test a few resets
for i in range(3):
    env.reset()
    current_edge = env.current_edge
    tile_id = f"tile_{current_edge}.png"
    
    print(f"   Reset {i+1}: Edge = {current_edge}")
    print(f"             Tile ID = {tile_id}")
    print(f"             Found in features: {tile_id in tile_features}")
    
    if tile_id in tile_features:
        feature = tile_features[tile_id]
        print(f"             Feature: shape={feature.shape}, range=[{feature.min():.3f}, {feature.max():.3f}]")
    else:
        print(f"             ❌ Tile ID not found - using zeros")

print(f"\n💡 POTENTIAL FIXES:")
print("1. Check if tile IDs match between features and edges")
print("2. Generate features for missing edges") 
print("3. Use alternative observation representation")
print("4. Implement geometric/positional features")