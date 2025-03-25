import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive maneuvers, specifically sliding tackles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success_coefficient = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'tackles': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['tackles'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["tackle_reward"][rew_index] = 0.0

            # Detect if sliding tackle was attempted
            if o['sticky_actions'][9] == 1:
                if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # opponent has the ball
                    d_ball_player = ((o['right_team'][o['ball_owned_player']][0] - o['left_team'][rew_index][0]) ** 2 +
                                     (o['right_team'][o['ball_owned_player']][1] - o['left_team'][rew_index][1]) ** 2) ** 0.5
                    if d_ball_player < 0.01:  # if player is very close to the ball carrier
                        components["tackle_reward"][rew_index] = self._tackle_success_coefficient
                        reward[rew_index] += components["tackle_reward"][rew_index]

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
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active
                info[f"sticky_actions_{i}"] = active
        return observation, reward, done, info
