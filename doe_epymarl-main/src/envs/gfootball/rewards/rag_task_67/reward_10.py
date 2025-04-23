import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a transitional play reward focusing on maintaining ball possession and executing passes
    under pressure from defense to attack. This rewards short passes, long passes, and dribbling.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.5
        self.dribble_reward = 0.3

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        """
        Enhance the reward function by adding extra rewards for passes and dribbling.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Check if it's a pass action (mock condition - to be replaced based on actual game logic)
            if o['sticky_actions'][0] == 1:
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]
            
            # Check if it's a dribble action (mock condition - to be replaced based on actual game logic)
            if o['sticky_actions'][9] == 1:
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]
            
        return reward, components
    
    def step(self, action):
        """
        Take a step using the given action.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions state for debug information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
