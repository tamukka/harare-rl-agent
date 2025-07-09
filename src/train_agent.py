import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from harare_env import HarareEnv
import torch as th

# Set up environment
def make_env():
    env = HarareEnv(
        edge_file="../data/harare_graph.gpkg",
        node_file="../data/harare_graph.gpkg",
        feature_file="../data/tile_features.pkl",
    )
    return env

env = DummyVecEnv([make_env])

# Create logs directory if it doesn't exist
os.makedirs("./logs/", exist_ok=True)

# Train PPO model with improved hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,  # Slightly higher learning rate
    n_steps=2048,        # More steps per update
    batch_size=128,      # Larger batch size
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    n_epochs=10,
    device="auto",
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network
    )
)

print("Starting training...")
print("Training will run for 1,000,000 timesteps...")
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("ppo_harare_rl_agent")
print("✅ Model saved to ppo_harare_rl_agent.zip")

# If you installed tensorboard, you can view logs with:
# tensorboard --logdir ./logs/