import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive and counterattack strategy reward component."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for rewarding defensive and counterattack positions
        self.defensive_threshold = 0.4   # Closer to own goal
        self.counterattack_threshold = 0.7  # Closer to opponent's goal
        self.defensive_reward = 0.05
        self.counterattack_reward = 0.1
        self.reset_defensive_positions()

    def reset(self):
        self.reset_defensive_positions()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reset_defensive_positions(self):
        self.recorded_defensive_positions = [False] * 4
        self.recorded_counterattack_positions = [False] * 4

    def reward(self, reward):
        """Reward for defensive plays and quick transitions to counterattack."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        reward_components = {"base_score_reward": reward.copy(),
                             "defensive_reward": [0.0] * len(reward),
                             "counterattack_reward": [0.0] * len(reward)}

        for i, o in enumerate(observation):
            player_x_position = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
            
            if not self.recorded_defensive_positions[i] and player_x_position < -self.defensive_threshold:
                reward_components["defensive_reward"][i] = self.defensive_reward
                self.recorded_defensive_positions[i] = True

            if not self.recorded_counterattack_positions[i] and player_x_position > self.counterattack_threshold:
                reward_components["counterattack_reward"][i] = self.counterattack_reward
                self.recorded_counterattack_positions[i] = True

            # Integrate the components into the main reward
            reward[i] += reward_components["defensive_reward"][i] + reward_components["counterattack_reward"][i]

        return reward, reward_components

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
