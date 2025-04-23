import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for shooting from central field positions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.power_shoot_reward = 1.0
        self.accuracy_bonus = 0.5
        self.central_zone = [-0.2, 0.2]  # The x-coordinate central zone range
        self.shoot_actions = {10}  # Hypothetical action index for shooting
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "shoot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if o['active'] in self.shoot_actions:
                # Check if player is in the central zone and has performed a shoot action
                if self.central_zone[0] <= o['ball'][0] <= self.central_zone[1]:
                    # Assuming accurate shoot leads to a potential goal
                    components["shoot_reward"][rew_index] += self.power_shoot_reward
                    if abs(o['ball_direction'][1]) < 0.1:  # Assuming lesser y movement is more "accurate"
                        components["shoot_reward"][rew_index] += self.accuracy_bonus
                    reward[rew_index] += sum(components["shoot_reward"][rew_index])
                
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
