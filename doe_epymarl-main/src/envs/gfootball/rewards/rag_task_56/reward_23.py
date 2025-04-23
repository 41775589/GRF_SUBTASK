import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards based on defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearing_successful = {}
        self.tackling_successful = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearing_successful = {}
        self.tackling_successful = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'clearing_successful': self.clearing_successful,
            'tackling_successful': self.tackling_successful
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearing_successful = from_pickle['CheckpointRewardWrapper']['clearing_successful']
        self.tackling_successful = from_pickle['CheckpointRewardWrapper']['tackling_successful']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearing_reward": [0.0] * len(reward),
            "tackling_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['game_mode'] in [3, 4]:
                # Counts successful clearings from free kicks and corners
                if o['ball_owned_team'] == 0 and self.clearing_successful.get(rew_index, False) == False:
                    components["clearing_reward"][rew_index] = 0.2
                    self.clearing_successful[rew_index] = True
            elif o['game_mode'] == 0:
                # Reset clearing successful flag during normal gameplay
                self.clearing_successful[rew_index] = False
            
            # Increase reward for successful tackles (no direct environment feedback, so we use a heuristic)
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0):
                if o['right_team_active'][o['active']] and not self.tackling_successful.get(rew_index, False):
                    components["tackling_reward"][rew_index] = 0.5
                    self.tackling_successful[rew_index] = True
            else:
                self.tackling_successful[rew_index] = False

            reward[rew_index] += components["clearing_reward"][rew_index] + components["tackling_reward"][rew_index]

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
