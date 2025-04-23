import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to incentivize maintaining ball control, strategic play, and effective ball distribution
    under pressure by the agent team. It increments dense rewards for maintaining possession, 
    passing in pressured situations, and exploiting open spaces efficiently.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tracking_player_control = []
        self.open_space_policy = []
        self.pass_frequency = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tracking_player_control = []
        self.open_space_policy = []
        self.pass_frequency = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'tracking_player_control': self.tracking_player_control,
            'open_space_policy': self.open_space_policy,
            'pass_frequency': self.pass_frequency
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tracking_player_control = from_pickle['CheckpointRewardWrapper']['tracking_player_control']
        self.open_space_policy = from_pickle['CheckpointRewardWrapper']['open_space_policy']
        self.pass_frequency = from_pickle['CheckpointRewardWrapper']['pass_frequency']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "control_reward": [0.0] * len(reward),
            "space_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:

                # Reward for maintaining control in pressured situations
                close_opponents = sum(np.linalg.norm(np.array(opponent) - np.array(obs['ball']), axis=1) < 0.1
                                      for opponent in obs['right_team'])
                if close_opponents >= 2:
                    components['control_reward'][i] += 0.1

                # Reward for moving to open spaces
                open_space = all(np.linalg.norm(np.array(player) - np.array(obs['ball']), axis=1) > 0.2
                                 for player in obs['left_team'] if player != obs['left_team'][obs['active']])
                if open_space:
                    components['space_reward'][i] += 0.2

                # Reward for successful passes
                if 'action' in obs and obs['action'] in [football_action_set.action_short_pass, football_action_set.action_long_pass]:
                    components['pass_reward'][i] += 0.1
                    self.pass_frequency.append((i, 1))

            reward[i] += sum(components[c][i] for c in components)

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
            if agent_obs['sticky_actions']:
                for i, action_status in enumerate(agent_obs['sticky_actions']):
                    if action_status:
                        self.sticky_actions_counter[i] += 1

        return observation, reward, done, info
