import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes efficient sliding tackles, rewarding precision and timing,
    specifically under high-pressure situations near the defending goal.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the sliding tackle reward
        self.sliding_tackle_reward = 0.5
        # Define the high-pressure zone close to the goal
        self.high_pressure_zone_threshold = -0.5
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if it's a high-pressure situation
            if o['ball'][0] < self.high_pressure_zone_threshold:
                # Check if the active player performs a sliding tackle
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # sliding tackle action index
                    # Check if it's against the opposing team's attempt towards goal
                    if o['ball_owned_team'] == 1 and o['ball_direction'][0] > 0:
                        components["tackle_reward"][rew_index] = self.sliding_tackle_reward
                        reward[rew_index] += components["tackle_reward"][rew_index]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
