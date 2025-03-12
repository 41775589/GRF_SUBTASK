import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_bonus = 0.1
        self.shoot_bonus = 0.3
        self.dribble_bonus = 0.2
        self.goal_score_bonus = 1.0
        # Track the last action to determine the type of reward to give
        self.last_action = None

    def reset(self):
        super(CheckpointRewardWrapper, self).reset()
        self.last_action = None

    def get_state(self, to_pickle):
        to_pickle['last_action'] = self.last_action
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_action = from_pickle.get('last_action', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward[:], 
                      "pass_bonus": [0.0] * len(reward),
                      "shoot_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward),
                      "goal_score_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if reward[rew_index] > 0:  # A goal has been scored
                components["goal_score_bonus"][rew_index] = self.goal_score_bonus
                reward[rew_index] += components["goal_score_bonus"][rew_index]

            # Checking sticky actions to give bonuses for passing, shooting, and dribbling
            if o['sticky_actions'][6] == 1:  # shoot
                components["shoot_bonus"][rew_index] = self.shoot_bonus
            elif o['sticky_actions'][5] == 1:  # long pass
                components["pass_bonus"][rew_index] = self.pass_bonus
            elif o['sticky_actions'][9] == 1:  # dribble
                components["dribble_bonus"][rew_index] = self.dribble_bonus

            reward[rew_index] += (components["shoot_bonus"][rew_index] + 
                                  components["pass_bonus"][rew_index] + 
                                  components["dribble_bonus"][rew_index])

        return reward, components

    def step(self, action):
        self.last_action = action
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
