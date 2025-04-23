import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds task-specific reward for high passes and crossings."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Define crossing and high pass zones based on y-position and ball direction
        self.crossing_thresholds = [0.75, -0.75]  # consider crossing in these y-ranges near the goals
        self.high_pass_min_height = 0.1  # minimum z position for a high pass

        # Rewards
        self.high_pass_reward = 0.3
        self.crossing_reward = 0.5

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
                      "high_pass_reward": [0.0] * len(reward),
                      "crossing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_height = o['ball'][2]
            ball_y = o['ball'][1]

            # Check for high passes
            if ball_height > self.high_pass_min_height:
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Check for crossings
            if abs(ball_y) >= self.crossing_thresholds[0] or abs(ball_y) <= self.crossing_thresholds[1]:
                components["crossing_reward"][rew_index] = self.crossing_reward
                reward[rew_index] += components["crossing_reward"][rew_index]

        return reward, components

    def step(self, action):
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
