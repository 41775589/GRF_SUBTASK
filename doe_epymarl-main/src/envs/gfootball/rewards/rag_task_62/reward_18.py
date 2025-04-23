import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that specializes in finishing techniques, focusing on optimizing shooting angles and timing
    under high-pressure scenarios near the goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._angle_threshold = 0.5  # Threshold for checking if the player is facing the goal
        self._pressure_factor = 0.2  # Factor for penalizing when opponents are close

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Adjust reward based on the player's shooting angle and opponents' proximity.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "angle_reward": [0.0] * len(reward), "pressure_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            obs = observation[i]

            # Calculate angle to goal
            goal_direction = [1, 0] if obs['active'] in obs['left_team_active'] else [-1, 0]
            player_pos = obs['left_team'][obs['active']] if obs['active'] in obs['left_team_active'] else obs['right_team'][obs['active']]
            player_direction = obs['left_team_direction'][obs['active']] if obs['active'] in obs['left_team_active'] else obs['right_team_direction'][obs['active']]

            facing_goal = np.dot(goal_direction, player_direction) > self._angle_threshold
            if facing_goal:
                components["angle_reward"][i] = 0.5
            
            # Calculate pressure from opponents
            opponents = obs['right_team'] if obs['active'] in obs['left_team_active'] else obs['left_team']
            distances_to_opponents = np.linalg.norm(opponents - player_pos, axis=1)
            close_opponents = np.sum(distances_to_opponents < 0.1)  # Count opponents closer than threshold

            if close_opponents > 0:
                components["pressure_reward"][i] = -self._pressure_factor * close_opponents

            reward[i] += components["angle_reward"][i] + components["pressure_reward"][i]

        return reward, components

    def step(self, action):
        """
        Take a step using the wrapped environment and modify the reward.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
