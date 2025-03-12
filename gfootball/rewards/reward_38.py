import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward to encourage offensive play strategies."""

    def __init__(self, env):
        super().__init__(env)
        self._shoot_reward = 0.1
        self._dribble_reward = 0.05
        self._pass_reward = 0.07

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shoot_reward': self._shoot_reward,
            'dribble_reward': self._dribble_reward,
            'pass_reward': self._pass_reward
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self._shoot_reward = state_data['shoot_reward']
        self._dribble_reward = state_data['dribble_reward']
        self._pass_reward = state_data['pass_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shoot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o["game_mode"] == 6:  # Penalty
                components["shoot_reward"][rew_index] += self._shoot_reward
                reward[rew_index] += components["shoot_reward"][rew_index]

            if 'sticky_actions' in o:
                if o['sticky_actions'][9] == 1:  # Dribbling
                    components["dribble_reward"][rew_index] += self._dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

            if 'game_mode' in o and o['game_mode'] in (1, 4, 5):  # Pass Types: KickOff, FreeKick, ThrowIn
                components["pass_reward"][rew_index] += self._pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
