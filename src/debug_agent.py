import numpy as np
import matplotlib.pyplot as plt
from harare_env import HarareEnv
from stable_baselines3 import PPO
import random
import pandas as pd

class AgentDiagnostics:
    def __init__(self, model_path="ppo_harare_rl_agent"):
        self.env = HarareEnv(
            edge_file="../data/harare_graph.gpkg",
            node_file="../data/harare_graph.gpkg",
            feature_file="../data/tile_features.pkl"
        )
        try:
            self.model = PPO.load(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
        
    def analyze_environment(self):
        """Analyze the environment structure and reward system"""
        print("\n🔍 ENVIRONMENT ANALYSIS")
        print("=" * 50)
        
        # Environment info
        print(f"📊 Environment Details:")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        print(f"   Number of edges: {len(self.env.edges)}")
        print(f"   Number of nodes: {len(self.env.nodes)}")
        
        # Test multiple random resets
        print(f"\n🎲 Testing Environment Resets:")
        for i in range(5):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            print(f"   Reset {i+1}: Starting edge = {self.env.current_edge}")
            print(f"             Observation shape = {obs.shape}")
            print(f"             Observation range = [{obs.min():.3f}, {obs.max():.3f}]")
            
    def test_random_actions(self, num_episodes=10):
        """Test what happens with random actions"""
        print(f"\n🎲 RANDOM ACTION ANALYSIS ({num_episodes} episodes)")
        print("=" * 50)
        
        episode_lengths = []
        episode_rewards = []
        termination_reasons = []
        
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                action = random.randint(0, self.env.action_space.n - 1)
                step_result = self.env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if done:
                    if reward < 0:
                        termination_reasons.append("negative_reward")
                    else:
                        termination_reasons.append("other")
                    break
            
            if not done:
                termination_reasons.append("max_steps")
                
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)
            
            print(f"   Episode {episode+1}: {steps} steps, {total_reward:.2f} reward, ended by {termination_reasons[-1]}")
        
        print(f"\n📈 Random Action Summary:")
        print(f"   Avg episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"   Avg total reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Termination reasons: {pd.Series(termination_reasons).value_counts().to_dict()}")
        
        return episode_lengths, episode_rewards, termination_reasons
    
    def test_agent_vs_random(self, num_episodes=10):
        """Compare agent performance vs random actions"""
        if self.model is None:
            print("❌ No model available for comparison")
            return
            
        print(f"\n🤖 AGENT VS RANDOM COMPARISON ({num_episodes} episodes each)")
        print("=" * 50)
        
        # Test agent
        agent_results = self._test_policy(self.model, num_episodes, "Agent")
        
        # Test random
        random_results = self._test_random_policy(num_episodes, "Random")
        
        # Compare
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Agent avg reward: {np.mean(agent_results['rewards']):.2f} vs Random: {np.mean(random_results['rewards']):.2f}")
        print(f"   Agent avg steps: {np.mean(agent_results['steps']):.1f} vs Random: {np.mean(random_results['steps']):.1f}")
        
        return agent_results, random_results
    
    def _test_policy(self, model, num_episodes, name):
        """Test a specific policy"""
        results = {'rewards': [], 'steps': [], 'terminations': []}
        
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                action = int(action[0])
                
                step_result = self.env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if done:
                    if reward < 0:
                        results['terminations'].append("negative_reward")
                    else:
                        results['terminations'].append("other")
                    break
            
            if not done:
                results['terminations'].append("max_steps")
                
            results['rewards'].append(total_reward)
            results['steps'].append(steps)
            
            print(f"   {name} Episode {episode+1}: {steps} steps, {total_reward:.2f} reward")
        
        return results
    
    def _test_random_policy(self, num_episodes, name):
        """Test random policy"""
        results = {'rewards': [], 'steps': [], 'terminations': []}
        
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                action = random.randint(0, self.env.action_space.n - 1)
                
                step_result = self.env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if done:
                    if reward < 0:
                        results['terminations'].append("negative_reward")
                    else:
                        results['terminations'].append("other")
                    break
            
            if not done:
                results['terminations'].append("max_steps")
                
            results['rewards'].append(total_reward)
            results['steps'].append(steps)
            
            print(f"   {name} Episode {episode+1}: {steps} steps, {total_reward:.2f} reward")
        
        return results
    
    def analyze_action_space(self):
        """Analyze what actions are available"""
        print(f"\n🎯 ACTION SPACE ANALYSIS")
        print("=" * 50)
        
        # Test different starting positions
        action_counts = {i: 0 for i in range(self.env.action_space.n)}
        
        for test in range(10):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            u, v, key = self.env.current_edge
            neighbors = list(self.env.graph.successors(v))
            
            print(f"   Test {test+1}: Edge {u}->{v}, {len(neighbors)} neighbors available")
            
            # Test each action
            for action in range(self.env.action_space.n):
                if action < len(neighbors):
                    action_counts[action] += 1
                    print(f"      Action {action}: Valid (neighbor {neighbors[action]})")
                else:
                    print(f"      Action {action}: Invalid (no neighbor)")
        
        print(f"\n📊 Action Usage Summary:")
        for action, count in action_counts.items():
            print(f"   Action {action}: Used in {count}/10 positions")
    
    def check_reward_distribution(self):
        """Check what rewards are actually given"""
        print(f"\n💰 REWARD DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        rewards_seen = []
        
        for test in range(50):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            for step in range(5):  # Take a few steps
                action = random.randint(0, self.env.action_space.n - 1)
                step_result = self.env.step(action)
                
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                rewards_seen.append(reward)
                
                if done:
                    break
        
        reward_counts = pd.Series(rewards_seen).value_counts().sort_index()
        print(f"   Rewards observed: {reward_counts.to_dict()}")
        print(f"   Most common reward: {reward_counts.index[0]} ({reward_counts.iloc[0]} times)")
        print(f"   Reward range: [{min(rewards_seen):.1f}, {max(rewards_seen):.1f}]")
        
        return rewards_seen

if __name__ == "__main__":
    diagnostics = AgentDiagnostics()
    
    # Run all diagnostics
    diagnostics.analyze_environment()
    diagnostics.test_random_actions(num_episodes=5)
    diagnostics.analyze_action_space()
    diagnostics.check_reward_distribution()
    
    # Compare agent vs random if model is available
    if diagnostics.model is not None:
        diagnostics.test_agent_vs_random(num_episodes=5)
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Check if the environment is too difficult")
    print("2. Verify reward structure makes sense")
    print("3. Consider reward shaping or curriculum learning")
    print("4. Check if observations contain useful information")
    print("5. Verify training actually improved the policy")