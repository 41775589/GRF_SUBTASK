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
            if 'ball_owned_player' in o and 'ball_owned_team' in o and o['ball_owned_player'] == rew_index and o['ball_owned_team'] == 1:
                components["checkpoint_reward"][rew_index] = 0.1  # Reward for being in ball possession as a defender
                reward[rew_index] += 1 * components["checkpoint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
