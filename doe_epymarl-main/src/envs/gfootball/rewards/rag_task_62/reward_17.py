import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically rewards players for optimizing shooting angles and timing under high-pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.angle_treshold = 0.1  # Define a threshold for a 'good' shooting angle
        self.pressure_threshold = 0.2  # Define a threshold for high-pressure situation
        self.goal_y_range = [-0.044, 0.044]  # Goal range on y-axis where scoring is possible

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Example, adjust as needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "angle_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        for i in range(len(reward)):
            o = observation[i]
            
            # Compute shooting angle and assess
            goal_midpoint = 1  # X-coordinate midpoint of the goal
            player_x, player_y = o['right_team'][o['active']]
            shooting_angle = abs(player_y / (goal_midpoint - player_x)) if player_x != goal_midpoint else float('inf')
            
            # Compute pressure
            # High pressure is assumed when one or more opponent players are close (within the pressure_threshold)
            opponent_distances = np.sqrt(np.sum(np.square(o['left_team'] - np.array([player_x, player_y])), axis=1))
            pressure = np.any(opponent_distances < self.pressure_threshold)
            
            # Reward for good shooting angles
            if abs(shooting_angle) < self.angle_treshold and o['ball_owned_team'] == 1:
                components["angle_reward"][i] = 0.3  # Tunable parameter for importance of angle reward

            # Reward for shooting under pressure
            if pressure and o['ball_owned_team'] == 1:
                components["pressure_reward"][i] = 0.5  # Tunable parameter for importance of pressure reward

            # Combine rewards
            reward[i] += components["angle_reward"][i] + components["pressure_reward"][i]

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
