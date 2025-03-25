import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on wide midfield responsibilities, focusing on high passes and wide positioning to stretch the defense."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Variables for handling high passes and position tracking
        self.high_pass_reward = 0.05
        self.positioning_reward = 0.02
        self.wide_flank_position_threshold = 0.7  # Position threshold to be considered wide on the y-axis

    def reset(self):
        """ Resets the sticky action counter on environment reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ State retrieval functionality to handle wrapper's state data. """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restores state data for the environment and this wrapper. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'], dtype=int)
        return from_pickle

    def reward(self, reward):
        """
        Adjusts the reward based on midfield wide opener responsibilities.
        Adds bonus rewards for effective high passes, positioning on the flank, and supporting lateral transitions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for idx, (o, rew) in enumerate(zip(observation, reward)):
            # Check for high pass action
            if o['sticky_actions'][9]:  # Assuming index 9 is for high pass action
                components["high_pass_reward"][idx] = self.high_pass_reward
                reward[idx] += components["high_pass_reward"][idx]

            # Reward for positioning on the right or left flank
            if abs(o['right_team'][idx][1]) >= self.wide_flank_position_threshold or \
               abs(o['left_team'][idx][1]) >= self.wide_flank_position_threshold:
                components["positioning_reward"][idx] = self.positioning_reward
                reward[idx] += components["positioning_reward"][idx]

        return reward, components

    def step(self, action):
        """
        Obtain observations and rewards from environment, apply reward modifications, and provide agent information.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
