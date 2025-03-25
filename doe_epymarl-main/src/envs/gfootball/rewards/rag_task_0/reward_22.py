import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for offensive football strategies including
    shooting, dribbling, and passing."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.2
        self.shoot_reward = 0.5
        self.dribble_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if 'sticky_actions' in o:
                if o['sticky_actions'][6]:  # Pass action
                    components['pass_reward'][rew_index] = self.pass_reward
                if o['sticky_actions'][7]:  # Shoot action
                    components['shoot_reward'][rew_index] = self.shoot_reward
                if o['sticky_actions'][9]:  # Dribble action
                    components['dribble_reward'][rew_index] = self.dribble_reward

                reward[rew_index] += (components['pass_reward'][rew_index] +
                                      components['shoot_reward'][rew_index] +
                                      components['dribble_reward'][rew_index])
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
