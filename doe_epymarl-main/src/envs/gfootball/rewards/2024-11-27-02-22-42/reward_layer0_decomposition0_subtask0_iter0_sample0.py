import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on successful offensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward = 0.2
        self.shot_reward = 1.0
        self.dribble_reward = 0.1

    def reset(self):
        """Reset and return the initial observation from the environment."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores the state of this reward wrapper alongside the environment's state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state from the given state object."""
        from_pickle = self.env.set_state(state)
        # If any state info for wrapper stored, it should be retrieved here
        return from_pickle

    def reward(self, reward):
        """Enhance the reward function by focusing on offensive action achievements."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_reward": np.zeros(len(reward), dtype=float), 
                      "shot_reward": np.zeros(len(reward), dtype=float), 
                      "dribble_reward": np.zeros(len(reward), dtype=float)}
        
        for player_index, obs in enumerate(observation):
            # Adding reward for successful passes
            if obs['sticky_actions'][1] == 1 or obs['sticky_actions'][9] == 1:  # indices for pass types
                components["pass_reward"][player_index] += self.pass_reward
                reward[player_index] += self.pass_reward
            
            # Adding reward for successful shots
            if obs['game_mode'] == 3 and obs['ball_owned_player'] == obs['active']:  # index for game_mode shot
                components["shot_reward"][player_index] += self.shot_reward
                reward[player_index] += self.shot_reward
            
            # Adding reward for dribbles
            if obs['sticky_actions'][10] == 1:  # index for dribble action
                components["dribble_reward"][player_index] += self.dribble_reward
                reward[player_index] += self.dribble_reward
        
        return reward, components

    def step(self, action):
        """Step the environment, modify the reward using the special reward function, and return observables."""
        observation, reward, done, info = self.env.step(action)
        # Modify the reward as per custom specification
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        # Adding components to info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
