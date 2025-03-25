import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense sprint-based reward for improving defensive coverage."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking usage of sticky actions

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Extract the current state for pickling, including the sticky actions counter."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from unpickled data, restoring the sticky actions counter."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards based on the agent's use of sprint for better defensive positioning."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            if o['sticky_actions'][8] == 1:  # index 8 corresponds to the sprint action
                # Increase reward progressively based on frequency of sprint action
                components["sprint_reward"][i] = 0.02 * self.sticky_actions_counter[8]
                reward[i] += components["sprint_reward"][i]

            # Update sticky actions counter for sprint
            self.sticky_actions_counter[8] += int(o['sticky_actions'][8])

        return reward, components

    def step(self, action):
        """Take a step using the given action, modify the reward, and collect additional info."""
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
