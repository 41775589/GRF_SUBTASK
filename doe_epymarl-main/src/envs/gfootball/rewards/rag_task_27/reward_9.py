import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on defensive actions in football training."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercepted_ball = False
        self.tight_marking = False
        self._intercept_reward = 0.5
        self._marking_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercepted_ball = False
        self.tight_marking = False
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['intercepted_ball'] = self.intercepted_ball
        state['tight_marking'] = self.tight_marking
        return state

    def set_state(self, state):
        self.intercepted_ball = state['intercepted_ball']
        self.tight_marking = state['tight_marking']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": 0.0,
                      "marking_reward": 0.0}

        if not observation:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            if o['ball_owned_team'] == 1 and o['left_team_active'][o['active']]:
                reward[i] += self._intercept_reward
                components["intercept_reward"] += self._intercept_reward
                self.intercepted_ball = True

            opponent_positions = o['right_team']
            player_position = o['left_team'][o['active']]
            if np.any([np.linalg.norm(player_position - opp) < 0.05 for opp in opponent_positions]):
                reward[i] += self._marking_reward
                components["marking_reward"] += self._marking_reward
                self.tight_marking = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"reward_component_{key}"] = value
        return observation, reward, done, info
