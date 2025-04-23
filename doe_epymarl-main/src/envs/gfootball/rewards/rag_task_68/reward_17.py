import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function to encourage offensive gameplay
    including accurate shooting, effective dribbling, and varied passing techniques.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward_coeff = 0.5  # Reward multiplier for shooting attempts
        self.dribbling_reward_coeff = 0.2  # Reward multiplier for dribbling
        self.passing_reward_coeff = 0.3   # Reward multiplier for diverse passing
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the environment and clears the sticky actions counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Stores the sticky actions state. """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restores the sticky actions state from a pickle. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Enhances the base reward function by adding specific rewards 
        for shooting, dribbling, and various types of passes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]

            # Check for shooting attempts
            if ('ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0.1):
                # Assume shooting is happening if the ball speed is high
                components["shooting_reward"][idx] = self.shooting_reward_coeff
                
            # Check for dribbling - sticky actions 9 indicate dribbling
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:
                components["dribbling_reward"][idx] = self.dribbling_reward_coeff
            
            # Incentivize diverse passing by recognizing long or high passes
            if 'ball_direction' in o and (abs(o['ball_direction'][0]) > 0.5 or o['ball_direction'][2] > 0.1):
                components["passing_reward"][idx] = self.passing_reward_coeff

            reward[idx] += components["shooting_reward"][idx]
            reward[idx] += components["dribbling_reward"][idx]
            reward[idx] += components["passing_reward"][idx]

        return reward, components

    def step(self, action):
        """
        Processes a step in the environment by applying the action
        and extracting the resulting observation and reward.
        Also tracks sticky actions usage.
        """
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
