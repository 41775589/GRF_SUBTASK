import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on shot precision near the goal."""

    def __init__(self, env):
        super().__init__(env)
        # Define proximity zones in front of the goal for precise shooting training
        self.goal_zones = np.linspace(-0.044, 0.044, 5)  # Dividing the goal width into zones
        self.reward_for_goal_zone = 0.2  # Reward for shooting from each zone
        self.shots_tried = {zone: False for zone in self.goal_zones}

    def reset(self):
        # Reset shot tracking on each episode
        self.shots_tried = {zone: False for zone in self.goal_zones}
        return self.env.reset()

    def reward(self, reward):
        # Calculate dense reward
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "precision_shot_reward": [0.0] * len(reward)}

        for rew_index, agent_reward in enumerate(reward):
            o = observation[rew_index]
            ball_pos_y = o['ball'][1]  # Get the Y position of the ball
        
            # Check if the shot is close to the goal and within the specified zones
            if abs(o['ball'][0] - 1) < 0.1:  # Ball close to the opponent's goal line
                for zone in self.goal_zones:
                    if (abs(ball_pos_y - zone) < 0.044 / len(self.goal_zones)) and not self.shots_tried[zone]:
                        components["precision_shot_reward"][rew_index] += self.reward_for_goal_zone
                        self.shots_tried[zone] = True  # Mark zone as tried
        
            # Combine rewards
            reward[rew_index] += components["precision_shot_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Step environment and process rewards
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
