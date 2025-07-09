import gymnasium as gym
from gymnasium import spaces
import random
import geopandas as gpd
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import osmnx as ox
from sklearn.preprocessing import StandardScaler

class ImprovedHarareEnv(gym.Env):
    def __init__(self,
                 edge_file="../data/harare_graph.gpkg",
                 node_file="../data/harare_graph.gpkg",
                 feature_file="../data/tile_features.pkl"):
        super(ImprovedHarareEnv, self).__init__()

        # Load data (same as original)
        self.edges = gpd.read_file(edge_file, layer="edges")
        self.nodes = gpd.read_file(node_file, layer="nodes")

        # Prepare nodes
        if 'osmid' in self.nodes.columns:
            self.nodes = self.nodes.set_index('osmid')
        elif 'id' in self.nodes.columns:
            self.nodes = self.nodes.set_index('id')
        else:
            self.nodes.reset_index(drop=True, inplace=True)
            self.nodes.index.name = 'osmid'

        self.nodes.index = self.nodes.index.astype(int)

        # Prepare edges
        if not all(col in self.edges.columns for col in ["u", "v", "key"]):
            raise ValueError("Edges file must contain columns: u, v, key")

        self.edges['u'] = self.edges['u'].astype(int)
        self.edges['v'] = self.edges['v'].astype(int)

        if 'key' not in self.edges.columns or self.edges['key'].isna().all():
            self.edges['key'] = 0

        # Remove invalid edges
        valid_nodes = set(self.nodes.index)
        mask = (self.edges['u'].isin(valid_nodes)) & (self.edges['v'].isin(valid_nodes))
        self.edges = self.edges[mask]
        self.edges = self.edges.set_index(["u", "v", "key"])
        self.edges = self.edges[~self.edges.index.duplicated(keep='first')]

        # Build graph
        try:
            self.graph = ox.graph_from_gdfs(self.nodes, self.edges)
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        # Load CNN features and create edge-to-feature mapping
        try:
            with open(feature_file, "rb") as f:
                self.tile_features = pickle.load(f)
            print(f"✅ Loaded {len(self.tile_features)} tile features")
        except:
            print("❌ Failed to load tile features, using geometric features only")
            self.tile_features = {}

        # Create improved observation system
        self._setup_observations()

        # Gym setup
        self.max_neighbors = 5
        self.action_space = spaces.Discrete(self.max_neighbors)
        # Increased observation space for richer features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.reset()

    def _setup_observations(self):
        """Create rich observation features for each edge"""
        print("🔧 Setting up improved observations...")
        
        self.edge_features = {}
        available_features = list(self.tile_features.values())
        
        # If we have tile features, use them; otherwise create geometric features
        if available_features:
            feature_dim = available_features[0].shape[0]
            print(f"   Using CNN features with {feature_dim} dimensions")
        
        # Create edge index mapping
        self.edge_list = list(self.edges.index)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edge_list)}
        
        # Compute features for each edge
        for edge_idx, edge in enumerate(self.edge_list):
            try:
                features = self._compute_edge_features(edge, edge_idx)
                self.edge_features[edge] = features
            except Exception as e:
                # Fallback to basic features
                self.edge_features[edge] = self._basic_edge_features(edge)
        
        print(f"✅ Created features for {len(self.edge_features)} edges")

    def _compute_edge_features(self, edge, edge_idx):
        """Compute rich features for an edge"""
        u, v, key = edge
        
        # 1. Use available tile feature or assign randomly
        if self.tile_features:
            # Assign tile features in a round-robin fashion
            tile_keys = list(self.tile_features.keys())
            assigned_tile = tile_keys[edge_idx % len(tile_keys)]
            cnn_feature = self.tile_features[assigned_tile][:10]  # Take first 10 dims
        else:
            cnn_feature = np.random.normal(0, 0.1, 10)  # Random baseline
        
        # 2. Geometric features
        try:
            row = self.edges.loc[edge]
            if hasattr(row.geometry, 'coords'):
                coords = list(row.geometry.coords)
                # Edge length (approximate)
                if len(coords) > 1:
                    start, end = coords[0], coords[-1]
                    length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                else:
                    length = 0.001
                
                # Edge direction (angle)
                if len(coords) > 1:
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    angle = np.arctan2(dy, dx)
                else:
                    angle = 0
                
                # Position features (normalized)
                center_x = np.mean([c[0] for c in coords])
                center_y = np.mean([c[1] for c in coords])
                
                geom_features = np.array([
                    length,
                    angle,
                    center_x / 32.0,  # Normalize roughly to Harare area
                    center_y / (-17.0),
                    len(coords)  # Number of coordinate points
                ])
            else:
                geom_features = np.zeros(5)
        except:
            geom_features = np.zeros(5)
        
        # 3. Graph connectivity features
        try:
            in_degree = self.graph.in_degree(v)
            out_degree = self.graph.out_degree(v)
            neighbors = len(list(self.graph.successors(v)))
            
            conn_features = np.array([
                in_degree,
                out_degree,
                neighbors,
                1.0 if neighbors > 0 else 0.0,  # Has valid actions
                min(neighbors, self.max_neighbors) / self.max_neighbors  # Action coverage
            ])
        except:
            conn_features = np.zeros(5)
        
        # Combine all features (10 + 5 + 5 = 20 dimensions)
        return np.concatenate([cnn_feature, geom_features, conn_features]).astype(np.float32)

    def _basic_edge_features(self, edge):
        """Fallback basic features"""
        u, v, key = edge
        
        # Use node IDs as pseudo-features (normalized)
        basic_features = np.array([
            (u % 1000) / 1000.0,
            (v % 1000) / 1000.0,
            key,
            random.random(),  # Some randomness
            random.random()
        ])
        
        # Pad to 20 dimensions
        padding = np.zeros(15)
        return np.concatenate([basic_features, padding]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Choose a starting edge that has at least one neighbor
        max_attempts = 100
        for _ in range(max_attempts):
            candidate_edge = random.choice(self.edge_list)
            u, v, key = candidate_edge
            neighbors = list(self.graph.successors(v))
            if len(neighbors) > 0:  # Must have valid actions
                self.current_edge = candidate_edge
                break
        else:
            # Fallback to any edge
            self.current_edge = random.choice(self.edge_list)
        
        self.path = [self.current_edge]
        self.steps_taken = 0
        return self._get_observation(), {}

    def step(self, action):
        u, v, key = self.current_edge
        neighbors = list(self.graph.successors(v))
        
        self.steps_taken += 1
        
        # Check if action is valid
        if action >= len(neighbors) or len(neighbors) == 0:
            # Invalid action - penalize and terminate
            return self._get_observation(), -10, True, True, {"reason": "invalid_action"}
        
        next_node = neighbors[action]
        
        try:
            # Get edge data
            edge_data = list(self.graph[v][next_node].values())[0]
            edge_id = edge_data.get('osmid')
            
            if isinstance(edge_id, list):
                edge_id = edge_id[0]
            
            # Find the edge in our edges dataframe
            edge_rows = self.edges[self.edges['osmid'] == edge_id]
            
            if edge_rows.empty:
                raise ValueError("Edge not found")
            
            self.current_edge = edge_rows.index[0]
            self.path.append(self.current_edge)
            
            # Improved reward system
            reward = self._calculate_reward()
            
            # Check termination conditions
            terminated = self._check_termination()
            truncated = self.steps_taken >= 50  # Episode length limit
            
        except Exception as e:
            # Navigation error
            reward = -5
            terminated = True
            truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {}

    def _calculate_reward(self):
        """Improved reward calculation"""
        u, v, key = self.current_edge
        neighbors = list(self.graph.successors(v))
        
        # Base reward for successful navigation
        base_reward = 1.0
        
        # Bonus for having many options (exploration)
        exploration_bonus = len(neighbors) * 0.1
        
        # Penalty for revisiting edges
        revisit_penalty = -0.5 if self.current_edge in self.path[:-1] else 0
        
        # Small penalty for long episodes (efficiency)
        efficiency_penalty = -0.01 * self.steps_taken
        
        total_reward = base_reward + exploration_bonus + revisit_penalty + efficiency_penalty
        
        return total_reward

    def _check_termination(self):
        """Check if episode should terminate"""
        u, v, key = self.current_edge
        neighbors = list(self.graph.successors(v))
        
        # Terminate if no valid actions
        if len(neighbors) == 0:
            return True
        
        # Terminate if stuck in a loop
        if len(self.path) > 5 and self.current_edge in self.path[-5:-1]:
            return True
        
        return False

    def _get_observation(self):
        """Get rich observation for current edge"""
        if self.current_edge in self.edge_features:
            return self.edge_features[self.current_edge]
        else:
            # Fallback
            return self._basic_edge_features(self.current_edge)

    def render(self, mode='human'):
        u, v, key = self.current_edge
        neighbors = list(self.graph.successors(v))
        print(f"🚗 Edge: {u}->{v}, {len(neighbors)} neighbors, Step: {self.steps_taken}")

# Test the improved environment
if __name__ == "__main__":
    print("🧪 TESTING IMPROVED ENVIRONMENT")
    print("=" * 50)
    
    env = ImprovedHarareEnv()
    
    # Test observations
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"   Non-zero elements: {np.count_nonzero(obs)}/{len(obs)}")
    
    # Test a few steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"   Step {step+1}: Action={action}, Reward={reward:.2f}, Done={terminated or truncated}")
        
        if terminated or truncated:
            break
    
    print(f"✅ Test completed: {step+1} steps, {total_reward:.2f} total reward")