import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper that focuses on shooting practice, encouraging the agent to practice
    shooting from centrally aligned field positions with high accuracy and power.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_x_position = 0  # Position of the goal horizontally
        self.central_y_threshold = 0.15  # Ball must be within this y range to be considered central
        self.shooting_distance_threshold = 0.3  # Distance from goal to be considered a shooting chance
        self.shooting_power_threshold = 8  # Minimum power to be considered a powerful shot

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            if 'ball' not in o and 'ball_owned_team' not in o:
                continue

            # Calculate distance to goal along the x-axis
            goal_distance = np.abs(self.goal_x_position - o['ball'][0])
            is_central = np.abs(o['ball'][1]) <= self.central_y_threshold

            # Check if the shot is powerful
            ball_speed = np.linalg.norm(o['ball_direction'][:2])  # First two coordinates are x, y speeds
            is_powerful_shot = ball_speed >= self.shooting_power_threshold

            # Calculate shooting reward when conditions align
            if is_central and goal_distance <= self.shooting_distance_threshold and is_powerful_shot:
                shooting_reward = 1.0
            else:
                shooting_reward = 0.0
            
            reward[index] += shooting_reward

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
