import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from improved_env import ImprovedHarareEnv
import torch as th

def make_env():
    """Create environment with improved observations"""
    env = ImprovedHarareEnv(
        edge_file="../data/harare_graph.gpkg",
        node_file="../data/harare_graph.gpkg",
        feature_file="../data/tile_features.pkl",
    )
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

# Create logs directory
os.makedirs("./logs/", exist_ok=True)

print("🚀 TRAINING IMPROVED HARARE RL AGENT")
print("=" * 50)
print("✅ Environment created with rich 20D observations")
print("✅ Improved reward system with exploration bonuses")
print("✅ Better termination conditions")

# Create improved PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.02,  # Higher entropy for more exploration
    clip_range=0.2,
    n_epochs=10,
    device="auto",
    policy_kwargs=dict(
        net_arch=[128, 128, 64],  # Smaller network for 20D input
        activation_fn=th.nn.ReLU
    )
)

print("\n🎯 Training Configuration:")
print(f"   Observation space: {env.observation_space}")
print(f"   Action space: {env.action_space}")
print(f"   Learning rate: {model.learning_rate}")
print(f"   Network architecture: [128, 128, 64]")
print(f"   Entropy coefficient: {model.ent_coef}")

# Create callbacks for better training monitoring
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="improved_ppo_harare"
)

# Start training
print("\n🏃‍♂️ Starting training for 500K timesteps...")
print("📊 Monitor progress with: tensorboard --logdir ./logs/")

model.learn(
    total_timesteps=500_000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
model.save("improved_ppo_harare_agent")
print("\n✅ Training completed!")
print("💾 Model saved as 'improved_ppo_harare_agent.zip'")

# Test the trained model
print("\n🧪 TESTING TRAINED MODEL")
print("=" * 30)

test_env = make_env()
obs, info = test_env.reset()

total_reward = 0
steps = 0
max_steps = 50

print("🎮 Running test episode...")

while steps < max_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    
    total_reward += reward
    steps += 1
    
    print(f"  Step {steps}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
    
    if terminated or truncated:
        break

print(f"\n📊 Test Results:")
print(f"   Episode length: {steps} steps")
print(f"   Total reward: {total_reward:.2f}")
print(f"   Average reward: {total_reward/steps:.2f}")
print(f"   Success: {'✅' if steps >= 10 and total_reward > 0 else '❌'}")

print(f"\n🎯 Next Steps:")
print("1. Run 'python test_improved_agent.py' to evaluate performance")
print("2. Launch dashboard to see improved visualizations")
print("3. Compare with previous model performance")