import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards associated with offensive football gameplay skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward weights for different actions
        self.shooting_reward = 2.0
        self.dribble_reward = 1.0
        self.pass_reward = 0.5
        # Action indices based on a hypothetical action set
        self.shoot_action = 9
        self.dribble_action = 10
        self.long_pass_action = 11
        self.high_pass_action = 12

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
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for player_index, o in enumerate(observation):
            active_sticky_actions = np.array(o['sticky_actions'])
            
            # Shooting reward
            if active_sticky_actions[self.shoot_action]:
                components["shoot_reward"][player_index] = self.shooting_reward
            
            # Dribbling reward
            if active_sticky_actions[self.dribble_action]:
                components["dribble_reward"][player_index] = self.dribble_reward

            # Passing reward, considering both long and high passes
            if active_sticky_actions[self.long_pass_action] or active_sticky_actions[self.high_pass_action]:
                components["pass_reward"][player_index] = self.pass_reward

            # Summing up the reward components
            total_component_reward = (components["shoot_reward"][player_index] +
                                      components["dribble_reward"][player_index] +
                                      components["pass_reward"][player_index])
            reward[player_index] += total_component_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
