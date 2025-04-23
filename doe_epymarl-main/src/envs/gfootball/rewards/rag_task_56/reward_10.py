import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adjusts rewards based on defensive capabilities:
    - Encourages the goalkeeper to stop shots and initiate plays
    - Encourages defenders to tackle effectively and retain ball possession"""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_stops = 0.3  # Reward for goalkeeper successful stops
        self.successful_tackles = 0.2  # Reward for successful tackles by defenders
        self.play_initiation = 0.1  # Reward for plays initiated by the goalkeeper

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        received_states = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = received_states['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_stops": np.zeros_like(reward),
            "defenders_tackles": np.zeros_like(reward),
            "play_initiations": np.zeros_like(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward goalkeeper for stopping shots
            if o['right_team_roles'][rew_index] == 0:  # Goalkeeper
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == rew_index:
                    components["play_initiations"][rew_index] = self.play_initiation
                    reward[rew_index] += self.play_initiation
                if o['ball_owned_team'] == 0:  # Ball is with opposing team
                    components["goalkeeper_stops"][rew_index] = self.successful_stops
                    reward[rew_index] += self.successful_stops

            # Reward defenders for tackles and retaining possession
            if o['right_team_roles'][rew_index] in [1, 2, 3, 4]:  # Defenders
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == rew_index:
                    components["defenders_tackles"][rew_index] = self.successful_tackles
                    reward[rew_index] += self.successful_tackles

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
