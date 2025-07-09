# Harare RL Agent - Urban Navigation with Reinforcement Learning

## Problem Definition

Urban navigation in African cities presents unique challenges due to complex road networks, dynamic traffic conditions, and limited real-time traffic data. Traditional GPS navigation systems often fail to capture the nuanced local knowledge required for optimal route planning in cities like Harare, Zimbabwe.

This project addresses the problem of **intelligent urban navigation** by training a reinforcement learning agent to navigate through Harare's road network using:
- **Graph-based environment**: Real OpenStreetMap road network data
- **Visual features**: Satellite imagery tiles processed through CNN for spatial context
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) algorithm for decision-making

## Solution Overview

The Harare RL Agent learns to navigate through the city by:
1. **Environment Modeling**: Converting Harare's road network into a graph-based RL environment
2. **Feature Engineering**: Extracting visual features from satellite tiles and combining them with geometric road properties
3. **Agent Training**: Using PPO to train an agent that makes navigation decisions at road intersections
4. **Real-time Visualization**: Interactive dashboard for monitoring agent performance and navigation paths

## Key Features

- =ú **Real Road Network**: Uses OpenStreetMap data for authentic Harare road topology
- =ð **Satellite Integration**: CNN-processed satellite imagery for spatial awareness
- >à **Deep RL**: PPO algorithm with customizable neural network architectures
- =Ê **Analytics Dashboard**: Streamlit-based real-time monitoring and visualization
- = **Incremental Training**: Checkpoint system for progressive model improvement
- <¯ **Multi-objective Rewards**: Balances exploration, efficiency, and valid navigation

## Technical Architecture

### Core Components

1. **Environment (`harare_env.py` & `improved_env.py`)**
   - Gymnasium-compatible RL environment
   - Graph-based state representation
   - Action space: Choose next road segment from available neighbors
   - Observation space: 20-dimensional feature vector combining CNN features, geometric properties, and graph connectivity

2. **Training System (`train_agent.py` & `train_improved_agent.py`)**
   - PPO implementation using Stable-Baselines3
   - Hyperparameter optimization for urban navigation
   - Tensorboard integration for training monitoring
   - Checkpoint saving for incremental improvements

3. **Data Pipeline**
   - `fetch_harare_roads.py`: Downloads OSM road network data
   - `fetch_satellite_tiles.py`: Retrieves satellite imagery tiles
   - `extract_file_features.py`: CNN feature extraction from satellite images
   - `save_harare_graph.py`: Processes and stores graph data

4. **Visualization & Analytics**
   - `dashboard.py`: Streamlit dashboard for real-time monitoring
   - `real_time_navigation.py`: Navigation system with path visualization
   - `visualize_path.py`: Static path visualization tools

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd harare-rl-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare data**
   ```bash
   cd src
   python fetch_harare_roads.py      # Download road network
   python fetch_satellite_tiles.py   # Download satellite tiles
   python extract_file_features.py   # Extract CNN features
   python save_harare_graph.py       # Process graph data
   ```

## Quick Start

### Training a New Agent

```bash
cd src
python train_improved_agent.py
```

### Testing the Agent

```bash
python test_improved_agent.py
```

### Launching the Dashboard

```bash
streamlit run dashboard.py
```

### Real-time Navigation

```bash
python real_time_navigation.py
```

## Data Files

- `data/harare_graph.gpkg`: Processed road network (nodes and edges)
- `data/harare_*.geojson`: Raw OpenStreetMap data
- `data/satellite_tiles/`: Satellite imagery tiles
- `data/tile_features.pkl`: CNN-extracted features from satellite images
- `data/harare_road_network.png`: Visualization of the road network

## Model Architecture

### Environment Features (20-dimensional observation space)
- **CNN Features (10 dims)**: Spatial context from satellite imagery
- **Geometric Features (5 dims)**: Road length, direction, position, coordinate points
- **Graph Features (5 dims)**: Node connectivity, valid actions, network topology

### PPO Configuration
- **Policy**: MLP with 256x256 hidden layers
- **Learning Rate**: 3e-4
- **Batch Size**: 128
- **Training Steps**: 2048 per update
- **Total Timesteps**: 500,000+ (incremental)

## Training Progress

The agent is trained incrementally with checkpoints saved every 50,000 steps:
- `checkpoints/improved_ppo_harare_50000_steps.zip`
- `checkpoints/improved_ppo_harare_100000_steps.zip`
- ... up to 500,000+ steps

Monitor training progress with:
```bash
tensorboard --logdir src/logs/
```

## Evaluation Metrics

- **Episode Reward**: Cumulative reward per navigation episode
- **Path Efficiency**: Ratio of optimal vs. actual path length
- **Valid Action Rate**: Percentage of valid navigation decisions
- **Exploration Coverage**: Unique road segments visited
- **Convergence Speed**: Training steps to reach performance threshold

## Performance Optimization

### Reward System
- **Base Reward**: +1.0 for successful navigation
- **Exploration Bonus**: +0.1 per available action
- **Revisit Penalty**: -0.5 for repeated edges
- **Efficiency Penalty**: -0.01 per step (encourages shorter paths)

### Termination Conditions
- No valid actions available
- Episode length exceeds 50 steps
- Agent stuck in navigation loop

## Applications

- **Urban Planning**: Analyze traffic flow patterns and bottlenecks
- **Route Optimization**: Develop locally-aware navigation systems
- **Traffic Management**: Understand network efficiency and congestion
- **Emergency Services**: Optimize response routes in urban environments

## Future Enhancements

- [ ] Multi-agent navigation scenarios
- [ ] Real-time traffic integration
- [ ] Weather and time-of-day factors
- [ ] Integration with mobile navigation apps
- [ ] Performance comparison with traditional algorithms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenStreetMap contributors for road network data
- Stable-Baselines3 for RL implementation
- OSMnx for graph processing utilities
- Streamlit for dashboard framework

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out to the project maintainers.

---

*Built with d for intelligent urban navigation in African cities*