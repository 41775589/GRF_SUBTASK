import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning and timing reward focusing on team synergy during possession changes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define parameters for the strategic positioning reward
        self.last_ball_position = np.zeros(3)
        self.possession_change_counter = 0
        self.positioning_reward_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_counter = 0
        self.last_ball_position = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position,
            'possession_change_counter': self.possession_change_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        self.possession_change_counter = from_pickle['CheckpointRewardWrapper']['possession_change_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Determine if there was a recent possession change
            if self.last_ball_position is not None and 'ball_owned_team' in o:
                if o['ball_owned_team'] != -1 and np.any(o['ball'] != self.last_ball_position):
                    self.possession_change_counter += 1
            
            # Reward based on strategic positioning after possession change
            if self.possession_change_counter > 0:
                distance_to_ball = np.linalg.norm(o['ball'] - o['left_team'][o['active']][:2])
                components["positioning_reward"][rew_index] = self.positioning_reward_factor / (1 + distance_to_ball)
                reward[rew_index] += components["positioning_reward"][rew_index]
            
            self.last_ball_position = o['ball'].copy()

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
