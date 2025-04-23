import gym
import numpy as np
class DribblingRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling skill improvement."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DribblingRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['DribblingRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy()}
        for i in range(len(reward)):
            o = observation[i]
            components.setdefault('dribbling_reward', [0.0] * len(reward))
            if o['sticky_actions'][9]:  # Checking if dribble action is active
                reward[i] += 0.01
                components['dribbling_reward'][i] += 0.01
            if o['sticky_actions'][8]:  # Checking if sprint action is active
                reward[i] += 0.02
                components['dribbling_reward'][i] += 0.02

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
