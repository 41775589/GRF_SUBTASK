import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamics reward to encourage midfield control and strategic transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_zones = {(0.33, 0.66): 'defensive', (-0.33, 0.33): 'midfield', (-0.66, -0.33): 'offensive'}
        self.player_data = {}  # To track positions and strategic transitions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_data = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.player_data
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        self.env.set_state(state)
        from_pickle = self.env.set_state(state)
        self.player_data = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for index, obs in enumerate(observation):
            self.update_player_data(index, obs)
            position_x = obs['left_team'][obs['active']][0] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']][0]

            # Players get rewarded for entering the midfield zone with the ball or maintaining position there
            components["midfield_control_reward"][index] = self.calculate_midfield_reward(index, position_x)

            reward[index] += components["midfield_control_reward"][index]

        return reward, components

    def calculate_midfield_reward(self, player_index, x_pos):
        reward_value = 0.0
        for zone in self.midfield_zones:
            if zone[0] <= x_pos < zone[1]:
                current_zone = self.midfield_zones[zone]
                last_zone = self.player_data[player_index].get('last_zone', None)
                if current_zone == 'midfield' and last_zone != 'midfield':
                    reward_value = 0.05  # Entering midfield is rewarded
                self.player_data[player_index]['last_zone'] = current_zone
        return reward_value

    def update_player_data(self, index, obs):
        if index not in self.player_data:
            self.player_data[index] = {}
        self.player_data[index].update({'last_position': obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]})

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
        return observation, reward, done, info
