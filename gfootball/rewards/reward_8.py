import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward linked to offensive football strategies including shooting accuracy, effective dribbling, and innovative passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.player_with_ball_last_step = -1

    def reset(self):
        self.player_with_ball_last_step = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'player_with_ball_last_step': self.player_with_ball_last_step}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.player_with_ball_last_step = from_pickle['CheckpointRewardWrapper']['player_with_ball_last_step']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_idx in range(len(reward)):
            o = observation[rew_idx]
            ball_owned_team = o['ball_owned_team']

            # Calculate shooting accuracy component
            if o["game_mode"] == 6:
                components["shooting_reward"][rew_idx] = 1.0
                reward[rew_idx] += components["shooting_reward"][rew_idx]

            # Calculate dribbling rewards when dribbling actions are active
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Dribble action is index 9
                components["dribble_reward"][rew_idx] = 0.5
                reward[rew_idx] += components["dribble_reward"][rew_idx]

            # Calculate passing reward considering high and long passes
            if self.player_with_ball_last_step != -1 and self.player_with_ball_last_step != o['active'] and ball_owned_team in [0, 1]:
                components["pass_reward"][rew_idx] = 0.3
                reward[rew_idx] += components["pass_reward"][rew_idx]

            if ball_owned_team in [0, 1]:
                self.player_with_ball_last_step = o['active']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
