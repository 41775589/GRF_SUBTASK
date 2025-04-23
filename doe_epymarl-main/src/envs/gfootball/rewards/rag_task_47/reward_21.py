import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for successful sliding tackles near our defensive third during counter-attacks and high-pressure situations.
    This encourages mastering the timing and positioning for sliding tackles.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_reward = 0.5
        self.tackle_attempt_penalty = -0.1
        self.defensive_third_threshold = -0.33  # Considering -1 to 1 as left to right on the x-axis
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "tackle_success_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_defensive_third = o['right_team'][o['active']][0] <= self.defensive_third_threshold
            has_ball = o['ball_owned_team'] == 1
            
            if is_defensive_third and has_ball:
                defensive_action_taken = o['sticky_actions'][5]  # Assumed index 5 to be sliding tackle
                
                if defensive_action_taken:
                    reward[rew_index] += self.tackle_attempt_penalty
                    
                    if np.random.rand() < 0.3:  # Assume a probabilistic model of success
                        reward[rew_index] += self.tackle_success_reward
                        components["tackle_success_reward"][rew_index] += self.tackle_success_reward

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
