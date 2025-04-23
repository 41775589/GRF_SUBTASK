import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents based on midfield dynamics control including positioning and coordination."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define areas (sectors) in midfield and their respective awards
        self.midfield_zones = [(-0.3, 0.3), (-0.2, 0.2)]
        self.midfield_rewards = [0.05, 0.1]  # rewards for ball control in these zones
    
    def reset(self):
        # Reset sticky actions counter for new episodes
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        # Allow environment to save its state
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        # Allow environment to restore its state
        return self.env.set_state(state)
    
    def reward(self, reward):
        # Modify the reward function to focus on maintaining midfield dynamics
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1 and abs(obs['ball'][0]) in self.midfield_zones[0]:
                # Reward for controlling the ball in the less central midfield area
                components['midfield_control_reward'][i] += self.midfield_rewards[0]
            if obs['ball_owned_team'] == 1 and abs(obs['ball'][0]) in self.midfield_zones[1]:
                # Higher reward for controlling the ball in the more central midfield area
                components['midfield_control_reward'][i] += self.midfield_rewards[1]
            
            # Integrate the midfield control rewards into the original reward
            reward[i] += components['midfield_control_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
