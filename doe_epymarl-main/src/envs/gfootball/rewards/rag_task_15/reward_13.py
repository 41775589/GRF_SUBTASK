import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards for mastering long passes, trajectory estimation, and targeting precision in football.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones for marking successful long pass regions on the field
        self.long_pass_regions = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1)]
        # Elect a passing reward
        self.passing_reward = 0.2
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        for index, rew in enumerate(reward):
            o = observation[index]
            if 'ball_owned_team' not in o:
                continue
            # Check if our team owns the ball
            if o['ball_owned_team'] == 1:  # assuming we're the right team
                # Calculate the distance the ball travels from the player
                ball_travel_dist = np.linalg.norm(o['ball_direction'][:2])
                # Check if the ball travel is within long pass distance
                for region_start, region_end in self.long_pass_regions:
                    if region_start <= ball_travel_dist < region_end:
                        components["pass_reward"][index] = self.passing_reward
                        reward[index] += components["pass_reward"][index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Accumulating component values and final reward to info for analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions, important for action analysis
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
