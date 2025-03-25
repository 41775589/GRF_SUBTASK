import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds sprint-based rewards for defensive coverage enhancements."""

    def __init__(self, env):
        super().__init__(env)
        # Keeping track of the frequency and effectiveness of sprint actions
        self.sprint_usage_counters = np.zeros(10, dtype=int)  # Arbitrary number of agents for demonstration

    def reset(self):
        """Reset the reward wrapper state for a new episode."""
        self.sprint_usage_counters.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include the wrapper's state in the pickleable state of the environment."""
        to_pickle['CheckpointRewardWrapper'] = self.sprint_usage_counters
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and extract the wrapper's state from the pickle."""
        from_pickle = self.env.set_state(state)
        self.sprint_usage_counters = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Customize rewards based on sprint usage for defensive positioning."""
        new_rewards = reward.copy()
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return new_rewards, {"base_score_reward": reward.copy()}

        components = {"base_score_reward": reward.copy(),
                      "sprint_usage_reward": [0.0] * len(reward)}

        for idx, single_observation in enumerate(observation):
            if 'sticky_actions' in single_observation:
                sprint_active = single_observation['sticky_actions'][8]  # Index 8 corresponds to 'action_sprint'
                if sprint_active:
                    if self.sprint_usage_counters[idx] < 10:  # Limit usage tracking to 10 for simplicity
                        self.sprint_usage_counters[idx] += 1
                        components["sprint_usage_reward"][idx] = 0.1  # Provide a reward for using sprint effectively
                        new_rewards[idx] += components["sprint_usage_reward"][idx]

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        return observation, reward, done, info
