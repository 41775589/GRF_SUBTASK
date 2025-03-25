import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for controlling midfield and strategic defense."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_rewards = np.zeros(5)  # 5 midfield horizontal zones
        self.defensive_blocks = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_rewards.fill(0)
        self.defensive_blocks = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = reward.copy()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "defensive_effort_reward": [0.0] * len(reward)}

        if observation is None:
            return modified_reward, components

        for i, obs in enumerate(observation):
            # Calculate midfield control reward based on ball position
            midfield_index = int((obs['ball'][0] + 1) // 0.4)  # Assuming field length normalized to [-1, 1]
            if 1 <= midfield_index <= 3:  # Middle three zones are considered critical midfield areas
                if obs['ball_owned_team'] == 0 and self.zone_rewards[midfield_index] == 0:
                    components["midfield_control_reward"][i] = 0.2
                    self.zone_rewards[midfield_index] = 1  # Mark zone as rewarded

            # Calculate defensive effort based on opponent's close proximity to the goal while not owning ball
            if obs['ball_owned_team'] == 1:
                opponent_distance_to_goal = min(np.linalg.norm(player_pos - np.array([1, 0])) for player_pos in obs['right_team'])
                if opponent_distance_to_goal < 0.1 and self.defensive_blocks == 0:
                    components["defensive_effort_reward"][i] += 0.3
                    self.defensive_blocks = 1  # Mark that defensive reward has been given

            modified_reward[i] += components["midfield_control_reward"][i] + components["defensive_effort_reward"][i]

        return modified_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
