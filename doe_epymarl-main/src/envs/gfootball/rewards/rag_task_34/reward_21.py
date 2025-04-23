import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on close-range attacking and dribbling against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_bonus_multiplier = 0.2
        self.shot_precision_bonus = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
    
    def reset(self):
        """Resets the environment and sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Include checkpoint data in the state to be pickled."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Retrieve state of the environment and wrapper from pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        """Compute the reward augmenting with dribbling and shot precision components."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'dribble_bonus': [0.0] * len(reward),
                      'shot_precision_bonus': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i, (obs, rew) in enumerate(zip(observation, reward)):
            # Reward for dribbling actions properly especially close to goal area
            if obs['sticky_actions'][9] == 1:  # Assuming index 9 is dribble action
                components['dribble_bonus'][i] = self.dribble_bonus_multiplier
                rew += components['dribble_bonus'][i]
                
            # Rewarding accurate shots - if in the goal area and shoots
            goal_area_thresh = 0.1
            if abs(obs['ball'][0]) > 1 - goal_area_thresh and obs['game_mode'] == 6:  # Assuming 6 as a shooting action
                components['shot_precision_bonus'][i] = self.shot_precision_bonus
                rew += components['shot_precision_bonus'][i]
        
            reward[i] = rew
        
        return reward, components
    
    def step(self, action):
        """Step through the environment applying the custom reward adjustments and tracking actions."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions count for informational purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
