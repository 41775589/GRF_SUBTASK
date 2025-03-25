import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones near the penalty area for defensive training
        self.defensive_zones = {
            'left_penalty': (-1.0, -0.2),  # Left penalty area on the X axis
            'left_close': (-1.0, -0.45),   # Close range left defense
            'right_penalty': (1.0, 0.2),   # Right penalty area on the X axis
            'right_close': (1.0, 0.45)     # Close range right defense
        }
        # Reward parameters
        self.zone_control_reward = 0.15
        self.defend_penalty_reward = 0.3

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
                      "defensive_positioning": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            x_pos = player_obs['left_team'][player_obs['active']][0]

            # Reward for controlling key defensive zones
            if x_pos < self.defensive_zones['left_penalty'][0]:
                components["defensive_positioning"][rew_index] += self.defend_penalty_reward
            elif x_pos < self.defensive_zones['left_close'][0]:
                components["defensive_positioning"][rew_index] += self.zone_control_reward

            reward[rew_index] += components["defensive_positioning"][rew_index]

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
