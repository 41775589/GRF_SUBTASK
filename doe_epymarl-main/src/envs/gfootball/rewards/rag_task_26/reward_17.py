import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds midfield dynamics specialized rewards, focusing on central and wide midfield contributions."""

    def __init__(self, env):
        super().__init__(env)
        self.central_midfield_threshold = 0.25  # Central area of the field longitudinally
        self.wide_midfield_threshold = 0.42  # Lateral width of the midfield considered "wide"
        self.central_midfield_reward = 0.05
        self.wide_midfield_reward = 0.03
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter

    def reset(self):
        """Reset the Gym environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state enhanced with wrapper specific data."""
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return to_pickle

    def set_state(self, state):
        """Set the state loading wrapper specific data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Compute reward with focus on midfield dynamics."""
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "central_midfield_reward": [0.0] * len(reward),
            "wide_midfield_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):

            # Calculating position-based rewards for central and wide midfielders
            x_pos = o['left_team'][o['active']][0]

            # Central Midfield Reward
            if abs(x_pos) <= self.central_midfield_threshold:
                components["central_midfield_reward"][rew_index] = self.central_midfield_reward
                reward[rew_index] += components["central_midfield_reward"][rew_index]

            # Wide Midfield Reward
            if abs(o['left_team'][o['active']][1]) >= self.wide_midfield_threshold:
                components["wide_midfield_reward"][rew_index] = self.wide_midfield_reward
                reward[rew_index] += components["wide_midfield_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take a step using the action and return the modified reward and environment information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
