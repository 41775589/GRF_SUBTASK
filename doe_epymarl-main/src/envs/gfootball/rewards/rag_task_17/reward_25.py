import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing midfield-wide responsibilities.
    It rewards mastering High Passes and good positioning to stretch the opponent's defense.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.wide_midfield_zones = [
            (-1.0, -0.42, -0.5, -0.2),  # left midfield area
            (0.5, -0.42, 1.0, -0.2)    # right midfield area
        ]
        self.high_pass_reward = 0.2  # Reward increment for successful high passes
        self.positioning_reward = 0.05  # Reward increment for maintaining correct positions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Initialize components for tracking detailed reward adjustments
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for making a high pass
            if 'sticky_actions' in o and o['sticky_actions'][6] == 1:
                reward[rew_index] += self.high_pass_reward
                components["high_pass_reward"][rew_index] = self.high_pass_reward
            
            # Confirm player positioning in midfield areas
            if o['ball_owned_team'] == (0 if o['active'] in o['left_team'] else 1):
                player_x, player_y = o['left_team'][o['active']] if o['active'] in o['left_team'] else o['right_team'][o['active']]
                for xmin, ymin, xmax, ymax in self.wide_midfield_zones:
                    if xmin <= player_x <= xmax and ymin <= player_y <= ymax:
                        reward[rew_index] += self.positioning_reward
                        components["position_reward"][rew_index] = self.positioning_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
