import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on ball control skills like short passes, long passes,
    and dribbles, which are crucial in transitioning from defense to offense.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.control_rewards = {
            'short_pass': 0.1,
            'long_pass': 0.2,
            'dribble': 0.05
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        state = self.env.set_state(state)
        return state

    def reward(self, reward):
        """
        Customize the reward function to include points for ball control skills.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Assuming a list with three players' rewards
        revised_reward = reward.copy()
        reward_components = {
            'base_score_reward': reward.copy(),
            'short_pass_reward': [0.0, 0.0, 0.0],
            'long_pass_reward': [0.0, 0.0, 0.0],
            'dribble_reward': [0.0, 0.0, 0.0]
        }

        for indx, o in enumerate(observation):
            sticky_actions = o.get('sticky_actions', [])
            if sticky_actions[6]:  # Assuming 'short_pass' is at index 6
                revised_reward[indx] += self.control_rewards['short_pass']
                reward_components['short_pass_reward'][indx] = self.control_rewards['short_pass']
            if sticky_actions[7]:  # Assuming 'long_pass' is at index 7
                revised_reward[indx] += self.control_rewards['long_pass']
                reward_components['long_pass_reward'][indx] = self.control_rewards['long_pass']
            if sticky_actions[9]:  # Assuming 'dribble' is at index 9
                revised_reward[indx] += self.control_rewards['dribble']
                reward_components['dribble_reward'][indx] = self.control_rewards['dribble']

        return revised_reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter for display
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
