import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that emphasizes offensive strategies, specifically shooting accuracy,
    dribbling skills, and performing varied pass types.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To keep track of sticky actions

        # Define coefficients for various actions to fine-tune the importance of each action
        self.shoot_coefficient = 1.0
        self.dribble_coefficient = 0.5
        self.pass_coefficient = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]

            # Reward for shooting if the action taken was a shoot action
            if obs['sticky_actions'][6]:  # Assuming index 6 corresponds to shooting
                components['shooting_reward'][idx] = self.shoot_coefficient

            # Reward for dribbling if the dribble action was active
            if obs['sticky_actions'][9]:  # Assuming index 9 corresponds to dribbling
                components['dribbling_reward'][idx] = self.dribble_coefficient

            # Additional reward for passing; considering long passes (high + long touches)
            if obs['sticky_actions'][1] or obs['sticky_actions'][3]:  # Assuming indices for long or high passes
                components['passing_reward'][idx] = self.pass_coefficient

            # Summing up all rewards
            reward[idx] += (
                components['shooting_reward'][idx] +
                components['dribbling_reward'][idx] +
                components['passing_reward'][idx]
            )

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
