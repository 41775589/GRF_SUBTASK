import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for a 'stopper' in football, focusing on man-marking, blocking, and defensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocking_bonus = 0.1
        self.marking_bonus = 0.05
        self.defensive_positioning_bonus = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "blocking_bonus": [0.0] * len(reward),
                      "marking_bonus": [0.0] * len(reward),
                      "defensive_positioning_bonus": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Enhance reward for blocking opponent's active moves by being in close marking positions.
            if o['active'] in o['right_team'] or o['active'] in o['left_team']:
                components["marking_bonus"][rew_index] = self.marking_bonus
                reward[rew_index] += components["marking_bonus"][rew_index]

            # Reward players for successful blocks
            if o['game_mode'] in [3, 4]:  # free kicks and corners potentially include blocks
                components["blocking_bonus"][rew_index] = self.blocking_bonus
                reward[rew_index] += components["blocking_bonus"][rew_index]

            # Add minor rewards for good defensive positioning regardless of ball possession
            components["defensive_positioning_bonus"][rew_index] = self.defensive_positioning_bonus
            if np.linalg.norm(o['ball'] - o['right_team'][o['active']]) > 0.5:  # rewarding distance maintenance
                reward[rew_index] += components["defensive_positioning_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # total reward for all agents
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
