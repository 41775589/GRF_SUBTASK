import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that enhances rewards for successful high passes, focusing on precise trajectory
    control and power assessment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # Prepare reward components
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }
        
        for rew_index, o in enumerate(observation):
            # Reward high passes
            if 'ball_direction' in o and o['ball_direction'][2] > 0.1:  # Significant upward trajectory
                components['high_pass_reward'][rew_index] = 0.2  # Constant reward increment to encourage high passes
                reward[rew_index] += components['high_pass_reward'][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Count sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        
        return observation, reward, done, info
