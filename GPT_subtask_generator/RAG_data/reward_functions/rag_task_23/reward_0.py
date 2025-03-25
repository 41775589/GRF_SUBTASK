import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive synergy reward focused on near-penalty area defense."""

    def __init__(self, env):
        super().__init__(env)
        self.penalty_area_threshold = 0.3  # Threshold for considering players in the defensive penalty area
        self.defensive_coordination_reward = 10.0  # Large reward for synergy in critical defense zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_synergy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        # Reinforce defensive coordination when multiple players are in the penalty area
        for team in ['left_team', 'right_team']:
            team_positions = observation[team]
            # Count players within the penalty area threshold
            defensive_players = np.sum(np.abs(team_positions[:, 0]) < self.penalty_area_threshold)
            if defensive_players > 1:  # More than one player in the defensive zone
                for i in range(len(reward)):
                    if (team == 'left_team' and i < len(reward) // 2) or (
                            team == 'right_team' and i >= len(reward) // 2):
                        components["defensive_synergy_reward"][i] += self.defensive_coordination_reward
                        reward[i] += components["defensive_synergy_reward"][i]

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
