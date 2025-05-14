import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on encouraging defensive positioning and maneuvers specific to intercepting 
    or tackling the opponent, especially around critical field areas like one's own goal zone."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_sprint_usage = np.zeros(2, dtype=float)  # Tracking total sprint for normalization
        self.total_slides = np.zeros(2, dtype=int)  # Count total slides
        self.goal_zone_threshold = -0.75  # Threshold for what is considered close to own goal (left side)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_sprint_usage.fill(0)
        self.total_slides.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "sprint_coefficient_reward": [0.0] * len(reward),
                      "slide_tackle_reward": [0.0] * len(reward)}

        # Check if observations are valid
        if observation is None:
            return reward, components

        for i, player_obs in enumerate(observation):
            # Position-based defensive rewards, specifically behind the defensive line or close to own goal
            player_x = player_obs['left_team'][player_obs['active']][0]  # x position of active player
            
            # Encourage defensive position deep in own half
            if player_x < self.goal_zone_threshold:
                components["defensive_positioning_reward"][i] += 0.05
            
            # Bonus for sprint actions in own half to quickly move to defensive positions
            if player_x < 0 and player_obs['sticky_actions'][8]:  # Sprinting active
                self.total_sprint_usage[i] += 0.1
                components["sprint_coefficient_reward"][i] = self.total_sprint_usage[i]
            
            # Bonus for sliding tackles in critical defensive areas
            if player_x < self.goal_zone_threshold and player_obs['sticky_actions'][9]:  # Sliding active
                self.total_slides[i] += 1
                components["slide_tackle_reward"][i] += 0.5 * self.total_slides[i]

            # Summing up the reward components
            reward[i] += sum(components[c][i] for c in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()  # Gather state information on sticky actions update
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[idx] += act

        return observation, reward, done, info
