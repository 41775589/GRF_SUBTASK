import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that reinforces the initiation of quick counter-attacks via Long Passes and transitioning from Stop-Moving to Sprint actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for debugging

    def reset(self):
        """Reset the environment and clear any trackers for sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Pickle the current environment state with any additional wrappers."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Unpickle the environment state into the current environment setup."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Adjust the rewards based on successful counter-attack actions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        for idx in range(len(reward)):
            obs = observation[idx]
            game_mode = obs['game_mode']
            sticky_actions = obs['sticky_actions']

            # Reward for successful long passes in counter-attack setups
            if game_mode == 0 and sticky_actions[9]:  # Assuming index 9 corresponds to Long Pass
                components["long_pass_reward"][idx] = 1.0

            # Reward for quick transitions from stop-moving to sprint
            if game_mode == 0 and sticky_actions[8] and not self.sticky_actions_counter[8]:
                components["transition_reward"][idx] = 1.0

            # Calculate final reward
            reward[idx] = sum([
                components["base_score_reward"][idx],
                components["long_pass_reward"][idx],
                components["transition_reward"][idx]
            ])

            # Update sticky actions counter for transitions
            self.sticky_actions_counter = sticky_actions

        return reward, components

    def step(self, action):
        """Apply an action to the environment, process rewards, and return the new observation and reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
