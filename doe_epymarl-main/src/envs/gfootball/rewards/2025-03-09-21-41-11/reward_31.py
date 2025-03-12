import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards focused on offensive skills in football such as accurate shooting, 
    dribbling and long/high passing strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribble_bonus = 0.1
        self.pass_bonus = 0.2
        self.shot_bonus = 0.5

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Logic to restore state if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "dribble_reward": [0.0] * len(reward), 
                      "pass_reward": [0.0] * len(reward), 
                      "shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            sticky_actions = o["sticky_actions"]

            # Reward for dribbling: check if dribble action is active
            if sticky_actions[9] == 1: 
                components["dribble_reward"][i] = self.dribble_bonus
                reward[i] += components["dribble_reward"][i]

            # Reward for passing: look for high or long passes depending on situation
            if o['game_mode'] in [3, 4]: # FreeKick or Corner
                components["pass_reward"][i] = self.pass_bonus
                reward[i] += components["pass_reward"][i]

            # Reward for accurate shooting
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0:  # Penalty and ball owned by left team (assuming)
                components["shot_reward"][i] = self.shot_bonus
                reward[i] += components["shot_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
