import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive actions."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._checkpoint_reward = 0.1
        self._collected_checkpoints = {0: 0, 1: 0}  # Initialize checkpoints for each agent
        self._num_checkpoints = 10

    def reset(self):
        self._collected_checkpoints = {0: 0, 1: 0}  # Reset checkpoints for each new episode
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        
        # Calculate checkpoint rewards for defensive actions
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # Check if the opponent team owns the ball
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5  # Calculate distance to the goal
                while (self._collected_checkpoints[rew_index] < self._num_checkpoints):
                    if d > 0.2:  # Set a threshold for collecting a checkpoint
                        break
                    components["checkpoint_reward"][rew_index] += self._checkpoint_reward
                    reward[rew_index] += 1.5 * components["checkpoint_reward"][rew_index]
                    self._collected_checkpoints[rew_index] += 1
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
