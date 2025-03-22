import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies: shooting, dribbling and passing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define the extra rewards for specific actions
        self.shooting_reward = 0.2
        self.pass_reward = 0.1
        self.dribble_reward = 0.05
        
    def reset(self):
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_sticky_actions = o['sticky_actions']  # Actions array

            # Check for action shoot
            if active_player_sticky_actions[9] == 1:  # Assuming index 9 is the shoot action
                components["shooting_reward"][rew_index] = self.shooting_reward
            # Check for action pass (long pass and high pass)
            if active_player_sticky_actions[0] == 1 or active_player_sticky_actions[1] == 1:  # Assuming indexes 0, 1 are for pass actions
                components["pass_reward"][rew_index] = self.pass_reward
            # Check for action dribble
            if active_player_sticky_actions[8] == 1:  # Assuming index 8 is the dribble action
                components["dribble_reward"][rew_index] = self.dribble_reward

            # Calculate final reward for the current player
            reward[rew_index] += (components["shooting_reward"][rew_index]
                                  + components["pass_reward"][rew_index]
                                  + components["dribble_reward"][rew_index])

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
