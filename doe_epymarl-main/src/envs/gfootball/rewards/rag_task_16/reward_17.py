import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes with precision."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_high_passes = 0
        self._high_pass_reward = 0.5
        # Threshold for considering the pass high, based on vertical ball speed
        self._vertical_speed_threshold = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_high_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['num_high_passes'] = self._num_high_passes
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        self._num_high_passes = from_pickle.get('num_high_passes', 0)
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_quality_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_index in range(len(reward)):
            obs = observation[agent_index]
            vertical_speed = obs['ball_direction'][2]  # Z-component speed of the ball

            # Reward for high passes
            if vertical_speed > self._vertical_speed_threshold:
                components["pass_quality_reward"][agent_index] = self._high_pass_reward
                reward[agent_index] += components["pass_quality_reward"][agent_index]
                self._num_high_passes += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions usage statistics in the info dictionary
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
