import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages efficient stamina management by rewarding minimal movement and sprint toggles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to tune the importance of resting and sprint toggles
        self.resting_reward = 0.01
        self.sprint_toggle_reward = 0.1

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores the current counts of sticky actions."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Loads the counts of sticky actions to maintain consistency."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Augments incoming reward based on agents' actions related to stamina management."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "resting_reward": [0.0] * len(reward),
                      "sprint_toggle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for resting: agent is inactive (neither moving nor sprinting)
            if np.array_equal(o['sticky_actions'][0:8], np.zeros(8, dtype=int)) and not o['sticky_actions'][8]:
                components["resting_reward"][rew_index] = self.resting_reward
                reward[rew_index] += components["resting_reward"][rew_index]

            # Reward for sprint toggle: check if sprint was toggled
            if o['sticky_actions'][8] != self.sticky_actions_counter[8]:
                components["sprint_toggle_reward"][rew_index] = self.sprint_toggle_reward
                reward[rew_index] += components["sprint_toggle_reward"][rew_index]

            # Updating the sticky actions count
            self.sticky_actions_counter[:] = o['sticky_actions']

        return reward, components

    def step(self, action):
        """Applies the agent's actions, computes a modified reward, and updates diagnostic info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
