import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive synergy in high-pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_good_positions = 0.1  # Reward for maintaining good defensive positions
        self._ball_intercept_reward = 1.0  # High reward for intercepting the ball in danger zone
        self._penalty_area_threshold = -0.3  # X coordinate threshold for the penalty area

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['defensive_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'] if o['designated'] in o['left_team_roles'] else o['right_team']
            
            # Calculate defensive good positions
            if any(pos[0] < self._penalty_area_threshold for pos in player_pos):  # Player in defensive third
                components["defensive_reward"][rew_index] = self._defensive_good_positions
                reward[rew_index] += components["defensive_reward"][rew_index]
            
            # Reward for intercepting the ball in defensive penalty area
            if o['ball'][0] < self._penalty_area_threshold and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["defensive_reward"][rew_index] += self._ball_intercept_reward
                reward[rew_index] += self._ball_intercept_reward

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
