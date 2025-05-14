import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive tasks like tactical tackling and strategic repositioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        self.total_defensive_actions = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        self.total_defensive_actions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['positional_rewards'] = self.positional_rewards
        to_pickle['total_defensive_actions'] = self.total_defensive_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positional_rewards = from_pickle['positional_rewards']
        self.total_defensive_actions = from_pickle['total_defensive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'defensive_rewards': [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_role = o['right_team_roles'][o['active']] if o['team'] == 'right' else o['left_team_roles'][o['active']]

            # Reward for successfully tackling and strategically repositioning
            if player_role in [1, 2, 3, 4]:  # Assuming these roles are defenders
                player_position = o['right_team'][o['active']] if o['team'] == 'right' else o['left_team'][o['active']]
                x_position = player_position[0]
                
                # Encourage moving towards opponent's attacking players or zones
                if x_position > 0.5:
                    x_reward = 0.1
                elif x_position > 0:
                    x_reward = 0.05
                else:
                    x_reward = 0.01

                self.positional_rewards[rew_index] = self.positional_rewards.get(rew_index, 0) + x_reward
                components['defensive_rewards'][rew_index] += x_reward

            reward[rew_index] = components['base_score_reward'][rew_index] + components['defensive_rewards'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
