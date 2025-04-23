import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on transition skills like passes and dribbles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.2
        self.dribbling_bonus = 0.1

    def reset(self):
        """Resets the environment and the internal sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Adjusts the given reward based on transition skills."""
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'passing_bonus': [0.0] * len(reward),
            'dribbling_bonus': [0.0] * len(reward)
        }
        
        if observation is None:
            return reward

        for rew_index, o in enumerate(observation):
            if 'sticky_actions' in o:
                if o['sticky_actions'][7] or o['sticky_actions'][8]:  # Action for passes (long/short)
                    components['passing_bonus'][rew_index] = self.passing_bonus
                    reward[rew_index] += components['passing_bonus'][rew_index]
                    self.sticky_actions_counter += o['sticky_actions']
                if o['sticky_actions'][9]:  # Action for dribbling
                    components['dribbling_bonus'][rew_index] = self.dribbling_bonus
                    reward[rew_index] += components['dribbling_bonus'][rew_index]
                    self.sticky_actions_counter += o['sticky_actions']

        return reward, components

    def step(self, action):
        """Steps through the environment, augmenting the rewards and extracting observations."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_val
        return observation, reward, done, info
