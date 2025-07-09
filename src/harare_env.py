import gymnasium as gym
from gymnasium import spaces
import random
import geopandas as gpd
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import osmnx as ox

class HarareEnv(gym.Env):
    def __init__(self,
                 edge_file="../data/harare_graph.gpkg",
                 node_file="../data/harare_graph.gpkg",
                 feature_file="../data/tile_features.pkl"):
        super(HarareEnv, self).__init__()

        # Load data
        self.edges = gpd.read_file(edge_file, layer="edges")
        self.nodes = gpd.read_file(node_file, layer="nodes")

        # Prepare nodes - OSMnx expects nodes to be indexed by node ID
        if 'osmid' in self.nodes.columns:
            self.nodes = self.nodes.set_index('osmid')
        elif 'id' in self.nodes.columns:
            self.nodes = self.nodes.set_index('id')
        else:
            # If no clear node ID column, reset index and use that
            self.nodes.reset_index(drop=True, inplace=True)
            self.nodes.index.name = 'osmid'

        # Ensure node index is integer (OSMnx typically expects this)
        self.nodes.index = self.nodes.index.astype(int)

        # Prepare edges - ensure u, v, key columns exist and are properly typed
        if not all(col in self.edges.columns for col in ["u", "v", "key"]):
            raise ValueError("Edges file must contain columns: u, v, key")

        # Ensure u and v are integers and match node indices
        self.edges['u'] = self.edges['u'].astype(int)
        self.edges['v'] = self.edges['v'].astype(int)

        # If key column doesn't exist or is all NaN, create it
        if 'key' not in self.edges.columns or self.edges['key'].isna().all():
            self.edges['key'] = 0

        # Remove any edges that reference non-existent nodes
        valid_nodes = set(self.nodes.index)
        mask = (self.edges['u'].isin(valid_nodes)) & (self.edges['v'].isin(valid_nodes))
        self.edges = self.edges[mask]

        # Set the multi-index AFTER cleaning the data
        self.edges = self.edges.set_index(["u", "v", "key"])

        # Remove duplicate edges
        self.edges = self.edges[~self.edges.index.duplicated(keep='first')]

        # Build graph from GeoDataFrames
        try:
            self.graph = ox.graph_from_gdfs(self.nodes, self.edges)
        except Exception as e:
            print(f"Error creating graph: {e}")
            print(f"Nodes shape: {self.nodes.shape}, index type: {type(self.nodes.index[0])}")
            print(f"Edges shape: {self.edges.shape}, index levels: {self.edges.index.names}")
            print(f"Sample nodes index: {self.nodes.index[:5].tolist()}")
            print(f"Sample edges index: {self.edges.index[:5].tolist()}")
            raise

        # Load CNN tile features
        with open(feature_file, "rb") as f:
            self.tile_features = pickle.load(f)

        # Gym setup
        self.max_neighbors = 5
        self.action_space = spaces.Discrete(self.max_neighbors)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Gymnasium reset method signature includes seed and options
        super().reset(seed=seed)
        self.current_edge = random.choice(list(self.edges.index))
        self.path = [self.current_edge]
        return self._get_observation(), {}

    def step(self, action):
        u, v, key = self.current_edge
        neighbors = list(self.graph.successors(v))

        if not neighbors:
            return self._get_observation(), -10, True, True, {}

        next_node = neighbors[min(action, len(neighbors) - 1)]

        try:
            # Get edge data
            edge_data = list(self.graph[v][next_node].values())[0]
            edge_id = edge_data.get('osmid')

            if isinstance(edge_id, list):
                edge_id = edge_id[0]

            # Find the edge in our edges dataframe
            edge_rows = self.edges[self.edges['osmid'] == edge_id]

            if edge_rows.empty:
                raise ValueError("Edge not found in edges GeoDataFrame.")

            self.current_edge = edge_rows.index[0]
            reward = 1
            terminated = False
            truncated = False

        except Exception as e:
            print(f"Error in step: {e}")
            reward = -1
            terminated = True
            truncated = False
        self.path.append(self.current_edge)
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        tile_id = f"tile_{self.current_edge}.png"
        feature = self.tile_features.get(tile_id, np.zeros(512))
        return feature.astype(np.float32)

    def render(self, mode='human'):
        print(f"🚗 Current edge: {self.current_edge}")