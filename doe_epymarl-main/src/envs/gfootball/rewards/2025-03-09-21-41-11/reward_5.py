import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic offensive play rewards including shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_bonus = 0.1
        self.shoot_bonus = 0.2
        self.dribble_bonus = 0.05

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "pass_reward": [0.0] * len(reward), "shoot_reward": [0.0] * len(reward), "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            if o['game_mode'] in (6, 2, 3):  # Modes corresponding to penalty, goal kick, and free kick.
                components['shoot_reward'][idx] = self.shoot_bonus
                reward[idx] += components['shoot_reward'][idx]
        
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Dribble action.
                components['dribble_reward'][idx] += self.dribble_bonus
                reward[idx] += components['dribble_reward'][idx]

            if 'sticky_actions' in o and (o['sticky_actions'][0] == 1 or o['sticky_actions'][4] == 1):  # Pass to the left or right.
                components['pass_reward'][idx] += self.pass_bonus
                reward[idx] += components['pass_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
