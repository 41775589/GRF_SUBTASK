import gym
import numpy as np
class GoalkeeperTrainingRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on specialized goalkeeper training."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.save_multiplier = 1.0
        self.pass_multiplier = 0.5
        self.goalkeeper_index = 0  # Assuming index 0 is always the goalkeeper for simplicity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['GoalkeeperTrainingRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['GoalkeeperTrainingRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['active'] == self.goalkeeper_index:
                if o['ball_owned_player'] == self.goalkeeper_index:
                    if any(o['sticky_actions'][:4]):  # Assuming first 4 actions are potential shot-stopping
                        components["save_reward"][rew_index] = self.save_multiplier
                    if o['sticky_actions'][8] == 1:  # Assuming action 8 is 'pass'
                        components["pass_reward"][rew_index] = self.pass_multiplier 
                
                # Update reward based on goalkeeper actions
                reward[rew_index] += (components["save_reward"][rew_index] + 
                                      components["pass_reward"][rew_index])

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
