import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 10
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {}
        return self.env.reset()

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            if reward[rew_index] == 1:
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward * (self._num_checkpoints - self._collected_checkpoints.get(rew_index, 0))
                reward[rew_index] = 1 * components["base_score_reward"][rew_index] + components["checkpoint_reward"][rew_index]
                self._collected_checkpoints[rew_index] = self._num_checkpoints
                continue
                
            if ('ball_owned_team' not in o or o['ball_owned_team'] == 0) and ('ball' in o):
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                while (self._collected_checkpoints.get(rew_index, 0) < self._num_checkpoints):
                    threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) * self._collected_checkpoints.get(rew_index, 0))
                    if d > threshold:
                        break
                    components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                    reward[rew_index] += 1.5 * components["checkpoint_reward"][rew_index]
                    self._collected_checkpoints[rew_index] = self._collected_checkpoints.get(rew_index, 0) + 1
                
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
    
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
