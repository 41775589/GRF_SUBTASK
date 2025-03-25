import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for playmaking skills in a football environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the reward wrapper state for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state of this reward wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of this reward wrapper from saved information."""
        from_pickle = self.env.set_state(state)
        # Restore any specific state if needed, here assumed none additional
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the agent behavior."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        # Assuming observation is a list with two elements (one for each agent), example provided structure reflects a single agent case.
        active_passing_reward = 0.1  # Reward for successful high or long passes
        dribbling_under_pressure_reward = 0.1  # Reward for maintaining possession under pressure
        effective_sprinting_reward = 0.05  # Reward for effective use of sprint in appropriate contexts

        for i in range(len(reward)):
            obs = observation[i]
            components[f"passing_reward_{i}"] = 0.0
            components[f"dribbling_reward_{i}"] = 0.0
            components[f"sprinting_reward_{i}"] = 0.0

            # Effective passing (simulate detecting high/long passes) 
            if obs['ball_direction'][0] > 0.1 and obs['sticky_actions'][8] == 1:  # Assuming index 8 might relate to long/high pass actions
                reward[i] += active_passing_reward
                components[f"passing_reward_{i}"] = active_passing_reward

            # Dribbling under pressure (just a simplistic check: close to an opponent)
            if obs['ball_owned_team'] == 0 and np.any(np.linalg.norm(obs['right_team'] - obs['ball'], axis=1) < 0.1):
                reward[i] += dribbling_under_pressure_reward
                components[f"dribbling_reward_{i}"] = dribbling_under_pressure_reward

            # Effective sprinting: sprint should not be constant, it's effective when used selectively
            if obs['sticky_actions'][8] == 1 and self.sticky_actions_counter[8] < 3:  # Assuming index 8 for sprinting
                reward[i] += effective_sprinting_reward
                components[f"sprinting_reward_{i}"] = effective_sprinting_reward

            self.sticky_actions_counter = obs['sticky_actions']

        return reward, components

    def step(self, action):
        """Processes the environment's step, modifying rewards and returning new observations and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
