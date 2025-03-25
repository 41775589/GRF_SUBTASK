import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for executing high passes effectively in a football environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for "high pass" success in qualitative terms.
        self.high_pass_success_threshold = 0.2  # Typically, this could be learned or adjusted.
        self.high_pass_bonus = 0.5  # Reward bonus for successful high passes.

    def reset(self):
        """Resets the sticky actions counter and environment states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for pickling, including wrapper-specific states."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from unpickling, including wrapper-specific states."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Computes the augmented reward function incorporating high pass bonuses."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Iterate over every agent's observation
        for i, obs in enumerate(observation):
            if 'ball_direction' in obs:
                # Example: Reward based on upward (positive y) and forward (positive x) ball direction with high 'z'
                ball_z_speed = obs['ball'][2]  # Access the z-component of ball's motion
                if ball_z_speed > self.high_pass_success_threshold:
                    # Checking for high pass completion could involve more conditions related to game state
                    components["high_pass_reward"][i] += self.high_pass_bonus
                    reward[i] += self.high_pass_bonus

        return reward, components

    def step(self, action):
        """Steps through the environment, applying the reward function modifications."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions info for understanding agent's behavior decisions
        last_obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in last_obs:
            for j, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active == 1:
                    self.sticky_actions_counter[j] += 1
        return obs, reward, done, info
