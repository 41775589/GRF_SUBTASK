import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense, defensive positional reward based on precise stopping and starting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define a counter for each player to track controlled stopping
        self.defensive_positioning_counter = np.zeros((2,), dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positioning_counter = np.zeros((2,), dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_counters': self.defensive_positioning_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positioning_counter = np.array(from_pickle['CheckpointRewardWrapper']['defensive_counters'])
        return from_pickle

    def reward(self, reward):
        # Initialize reward components
        components = {
            "base_score_reward": reward.copy(), 
            "defensive_position_reward": [0.0, 0.0]
        }
        
        # Fetch current observations from the environment
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Calculate defensive reward for stopping effectively
            if o['sticky_actions'][0] == 0:  # Indicates the 'action_idle' is active
                self.defensive_positioning_counter[rew_index] += 1
            
            if self.defensive_positioning_counter[rew_index] > 100:  # arbitrary threshold
                components["defensive_position_reward"][rew_index] = 0.2  # reward for maintaining good defensive position
                reward[rew_index] += components["defensive_position_reward"][rew_index]
                # Reset counter once rewarded
                self.defensive_positioning_counter[rew_index] = 0
        
        return reward, components

    def step(self, action):
        # Step the environment
        observation, reward, done, info = self.env.step(action)
        
        # Calculate custom reward components
        reward, components = self.reward(reward)
        
        # Append the calculated components to the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Return the observation, modified reward, done and info
        return observation, reward, done, info
