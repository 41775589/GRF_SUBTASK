import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper designed for enhancing the defensive capabilities of agents in a football simulation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.defender_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.defender_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['goalkeeper_rewards'] = self.goalkeeper_rewards
        to_pickle['defender_rewards'] = self.defender_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_rewards = from_pickle.get('goalkeeper_rewards', {})
        self.defender_rewards = from_pickle.get('defender_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_reward": [0.0] * len(reward),
                      "defender_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for the goalkeeper stopping a shot or initiating play
            if observation['right_team_roles'][rew_index] == 0:  # 0 is the index for a goalkeeper
                if 'game_mode' in o and o['game_mode'] == 6:  # 6 is the index for Penalty
                    self.goalkeeper_rewards[rew_index] = 1.0  # Basic good reward on shot-stopping scenario
                # Initiating play 
                elif o['action'] == 'long_pass':
                    self.goalkeeper_rewards[rew_index] += 0.1
                # Adjust final reward for GK
                components["goalkeeper_reward"][rew_index] = self.goalkeeper_rewards[rew_index]
                reward[rew_index] += components["goalkeeper_reward"][rew_index]

            # Reward for defenders performing a successful tackle or retaining possession
            if observation['right_team_roles'][rew_index] in [1, 2, 3]:  # Indices for defenders
                if 'last_action' in o and o['last_action'] == 'slide':
                    self.defender_rewards[rew_index] = 0.5  # Reward tackling
                if 'ball_owned_player' in o and o['ball_owned_player'] == rew_index:
                    self.defender_rewards[rew_index] += 0.1  # Ball retention
                # Adjust final reward for Defenders
                components["defender_reward"][rew_index] = self.defender_rewards[rew_index]
                reward[rew_index] += components["defender_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
