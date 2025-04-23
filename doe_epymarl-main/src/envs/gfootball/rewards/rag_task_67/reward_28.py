import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on passing and dribbling skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.2
        self.dribbling_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "passing_reward": [0.0] * len(reward), 
            "dribbling_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for pass action execution (either short or long pass)
            if 'sticky_actions' in o:
                if o['sticky_actions'][4] == 1 or o['sticky_actions'][5] == 1:  # Assuming 4 and 5 map to passing actions
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]
                    
            # Check for dribbling action execution
            if 'sticky_actions' in o:
                if o['sticky_actions'][9] == 1:  # Assuming 9 maps to dribbling action
                    components["dribbling_reward"][rew_index] = self.dribbling_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

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
