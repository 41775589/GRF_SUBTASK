import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for coordinating midfielders and strikers in offensive plays."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_strikers_cooperation = {}
        self.cooperation_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_strikers_cooperation = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "cooperation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = components["base_score_reward"][rew_index]

            if o['game_mode'] != 0:  # Only apply during normal play
                continue

            # Reward midfielders and strikers coordination
            if o['ball_owned_team'] == o['active'] and \
               o['active'] in [4, 5, 6, 7, 8, 9]:  # Assuming IDs for midfield and strikers
                ball_position_y = o['ball'][1]
                if not self.midfielder_strikers_cooperation.get(rew_index, False) and \
                   ball_position_y > 0.1:  # Position Y coordinate heuristic for forward play
                    components["cooperation_reward"][rew_index] += self.cooperation_reward
                    self.midfielder_strikers_cooperation[rew_index] = True

            reward[rew_index] = base_reward + components["cooperation_reward"][rew_index]

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
