import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defensive reward for mastering defensive responsiveness 
    and interception skills.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.defensive_checkpoints = {}
        self.num_defensive_zones = 5
        self.defensive_zone_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Adjust reward based on agent's ability to maintain defensive positions and 
        intercept the ball, especially in high-pressure scenarios.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_zone_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            distance_to_ball = np.linalg.norm(o['ball'] - o['left_team'][o['active']])
            defending_efficiency = 1 - min(distance_to_ball, 1)
            
            # Compute defensive reward based on the player's proximity to the ball
            # and defensive zone importance
            defensive_coefficient = (1.0 - (o['left_team'][o['active']][0] + 1) / 2)  # Convert range [-1, 1] to [0, 1]
            components["defensive_zone_reward"][rew_index] = self.defensive_zone_reward * defensive_coefficient * defending_efficiency

            reward[rew_index] += components["defensive_zone_reward"][rew_index]
            
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
