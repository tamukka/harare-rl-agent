import numpy as np
from improved_env import ImprovedHarareEnv
from stable_baselines3 import PPO
import os

def test_improved_agent():
    print("🧪 TESTING IMPROVED AGENT PERFORMANCE")
    print("=" * 50)
    
    # Create environment
    env = ImprovedHarareEnv(
        edge_file="../data/harare_graph.gpkg",
        node_file="../data/harare_graph.gpkg",
        feature_file="../data/tile_features.pkl"
    )
    
    # Try to load the latest model
    model_files = ["improved_ppo_harare_agent.zip", "ppo_harare_rl_agent.zip"]
    model = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = PPO.load(model_file)
                print(f"✅ Loaded model: {model_file}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
    
    if model is None:
        print("❌ No trained model found")
        return
    
    # Test multiple episodes
    results = {
        'episode_lengths': [],
        'episode_rewards': [],
        'success_episodes': 0,
        'avg_confidence': []
    }
    
    num_test_episodes = 10
    print(f"\n🎮 Running {num_test_episodes} test episodes...")
    
    for episode in range(num_test_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        confidences = []
        
        print(f"\n📊 Episode {episode + 1}:")
        
        while steps < max_steps:
            # Get action with confidence
            action, _states = model.predict(obs, deterministic=False)
            
            # Calculate confidence (simplified)
            action_probs = model.policy.predict_values(obs.reshape(1, -1))[0]
            confidence = np.max(action_probs) if hasattr(action_probs, '__len__') else 0.5
            confidences.append(confidence)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"   Step {steps}: Action={action}, Reward={reward:.2f}, Confidence={confidence:.3f}")
            
            if terminated or truncated:
                reason = info.get('reason', 'terminated' if terminated else 'truncated')
                print(f"   🏁 Episode ended: {reason}")
                break
        
        # Record results
        results['episode_lengths'].append(steps)
        results['episode_rewards'].append(total_reward)
        results['avg_confidence'].append(np.mean(confidences) if confidences else 0)
        
        # Success criteria: episode length > 5 and positive reward
        if steps > 5 and total_reward > 0:
            results['success_episodes'] += 1
            print(f"   ✅ Success! {steps} steps, {total_reward:.2f} reward")
        else:
            print(f"   ❌ Failed: {steps} steps, {total_reward:.2f} reward")
    
    # Calculate statistics
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 30)
    print(f"Success Rate: {results['success_episodes']}/{num_test_episodes} ({results['success_episodes']/num_test_episodes*100:.1f}%)")
    print(f"Avg Episode Length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print(f"Avg Episode Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"Avg Confidence: {np.mean(results['avg_confidence']):.3f} ± {np.std(results['avg_confidence']):.3f}")
    print(f"Best Episode: {max(results['episode_rewards']):.2f} reward, {max(results['episode_lengths'])} steps")
    
    # Compare with baseline
    print(f"\n🔄 COMPARISON WITH BASELINE")
    print("=" * 30)
    print(f"Previous Results: 0% success, -4.87 avg reward, 0.28 confidence")
    
    improvement_success = (results['success_episodes']/num_test_episodes*100) - 0
    improvement_reward = np.mean(results['episode_rewards']) - (-4.87)
    improvement_confidence = np.mean(results['avg_confidence']) - 0.28
    
    print(f"Improvements:")
    print(f"  Success Rate: +{improvement_success:.1f}% 📈")
    print(f"  Avg Reward: +{improvement_reward:.2f} 📈")
    print(f"  Confidence: +{improvement_confidence:.3f} 📈")
    
    return results

if __name__ == "__main__":
    results = test_improved_agent()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Continue training if performance needs improvement")
    print("2. Update dashboard to use improved environment")
    print("3. Run comparative analysis with visualization")
    print("4. Deploy improved model for real-time navigation")