import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a strategic play reward focusing on midfielders and strikers interplay.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielders_collected_points = {}
        self.strikers_collected_points = {}
        self.midfielder_reward_coefficient = 0.2
        self.striker_reward_coefficient = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielders_collected_points = {}
        self.strikers_collected_points = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'midfielders_collected_points': self.midfielders_collected_points,
            'strikers_collected_points': self.strikers_collected_points
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfielders_collected_points = from_pickle['CheckpointRewardWrapper']['midfielders_collected_points']
        self.strikers_collected_points = from_pickle['CheckpointRewardWrapper']['strikers_collected_points']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfielder_reward": [0.0] * len(reward),
            "striker_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Reward midfielders for successful ball delivery
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] in self.midfielders_collected_points:
                if obs['ball'][0] > 0.5:  # past midfield
                    if self.midfielders_collected_points.get(obs['ball_owned_player'], 0) < 1:
                        components["midfielder_reward"][i] = self.midfielder_reward_coefficient
                        reward[i] += components["midfielder_reward"][i]
                        self.midfielders_collected_points[obs['ball_owned_player']] = 1

            # Reward strikers for successful play finishing
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] in self.strikers_collected_points:
                if obs['ball'][0] > 0.8:  # in the final third
                    if self.strikers_collected_points.get(obs['ball_owned_player'], 0) < 1:
                        components["striker_reward"][i] = self.striker_reward_coefficient
                        reward[i] += components["striker_reward"][i]
                        self.strikers_collected_points[obs['ball_owned_player']] = 1

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
