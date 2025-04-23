import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specialized rewards for high passes and crossing
    from varying distances and angles to promote dynamic attacking plays.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distance and angle thresholds for high passes and crossings
        self.crossing_distance_threshold = 0.5  # Example threshold
        self.high_pass_angle_threshold_rad = np.pi / 4  # 45 degrees in radians
        self.reward_for_crossing = 0.2
        self.reward_for_high_pass = 0.1

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
                      "crossing_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Apply additional reward logic 
        for rew_index, _ in enumerate(reward):
            o = observation[rew_index]

            ball_pos = np.array(o['ball'][:2])  # Only consider x, y coordinates
            ball_direction = np.array(o['ball_direction'][:2])

            # Compute distances and angles relevant for crossing and high passes
            ball_speed = np.linalg.norm(ball_direction)
            if ball_speed == 0:
                continue  # Neglect static ball situation

            ball_travel_angle = np.arctan2(ball_direction[1], ball_direction[0])

            # Check if it's a crossing scenario
            if np.abs(ball_pos[0]) > self.crossing_distance_threshold:
                components["crossing_reward"][rew_index] = self.reward_for_crossing
                reward[rew_index] += components["crossing_reward"][rew_index]

            # Check if it's a high pass situation
            if np.abs(ball_travel_angle) >= self.high_pass_angle_threshold_rad:
                components["high_pass_reward"][rew_index] = self.reward_for_high_pass
                reward[rew_index] += components["high_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
