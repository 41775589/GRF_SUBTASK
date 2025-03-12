import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper for offensive football strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_reward_coefficient = 0.2
        self.shot_reward_coefficient = 0.3
        self.dribble_reward_coefficient = 0.1
        self.goal_reward_coefficient = 1.0  # Goal scoring has the highest reward
    
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
        components = {'base_score_reward': reward.copy(),
                      'pass_reward': [0.0] * len(reward),
                      'shot_reward': [0.0] * len(reward),
                      'dribble_reward': [0.0] * len(reward),
                      'goal_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, (rew, obs) in enumerate(zip(reward, observation)):
            if obs['game_mode'] == 6:  # Penalty Shot
                components['shot_reward'][i] += self.shot_reward_coefficient
            if obs['sticky_actions'][9] == 1:  # Dribbling
                components['dribble_reward'][i] += self.dribble_reward_coefficient
            if obs['game_mode'] == 3:  # Free Kick
                components['shot_reward'][i] += self.shot_reward_coefficient
            if obs['score'][0] > obs['score'][1]:  # Goal scored
                components['goal_reward'][i] += self.goal_reward_coefficient

            # Summing up the rewards from different components
            reward[i] += (components['pass_reward'][i] +
                          components['shot_reward'][i] +
                          components['dribble_reward'][i] +
                          components['goal_reward'][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
