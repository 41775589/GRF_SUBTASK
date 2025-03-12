import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'left_team_direction' in o and 'right_team_direction' in o:
                # Calculate the distance between the directions of the left and right team players
                distance = np.linalg.norm(o['left_team_direction'] - o['right_team_direction'])
                components["checkpoint_reward"][rew_index] = max(0, 1 - distance)  # Reward based on distance between team directions
                reward[rew_index] += 0.1 * components["checkpoint_reward"][rew_index]  # Adjust the coefficient

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
