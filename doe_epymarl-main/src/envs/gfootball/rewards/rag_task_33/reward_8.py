import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for long-range shots successfully taken outside the penalty box."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_range_shot_reward = 2.0  # Reward multiplier for successful long-range shots
        self.penalty_box_x_threshold = 0.7  # X-axis threshold for qualifying a long-range shot

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_range_shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Check for goal scored condition and ball position
            if base_reward == 1 and self._is_long_range_shot(o):
                components["long_range_shot_reward"][rew_index] = self.long_range_shot_reward
                reward[rew_index] += components["long_range_shot_reward"][rew_index]

        return reward, components

    def _is_long_range_shot(self, obs):
        """Check if the scoring shot was taken from outside the penalty box."""
        ball_pos = obs['ball'][0]  # Only consider x coordinate
        return abs(ball_pos) < self.penalty_box_x_threshold

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = float(sum(reward))
        for key, value in components.items():
            info[f"component_{key}"] = float(sum(value))
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
