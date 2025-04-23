import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes and crosses at different distances and angles."""

    def __init__(self, env, num_zones=5, pass_reward=0.1):
        super().__init__(env)
        self.num_zones = num_zones
        self.pass_reward = pass_reward
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_counters = np.zeros((2, num_zones), dtype=int)  # Using a 2x5 matrix for each player side

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_counters = np.zeros_like(self.pass_counters)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'pass_counters': self.pass_counters}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_counters = from_pickle['CheckpointRewardWrapper']['pass_counters']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation), "The number of observations must match the number of rewards."

        for r_idx, obs in enumerate(observation):
            if 'ball_owned_team' in obs and obs['ball_owned_team'] in [0, 1]:
                ball_owner = obs['ball_owned_team']
                zones_covered = min(int(obs['ball'][0] * self.num_zones + 0.5 * self.num_zones), self.num_zones - 1)
                # Determine the player's own team index in the pass_counters
                team_idx = 0 if ball_owner == obs['left_team'] else 1

                if self.pass_counters[team_idx, zones_covered] < 1:
                    distance = np.linalg.norm(np.array(obs['ball']) - np.array(obs['left_team' if team_idx == 0 else 'right_team']))
                    angle = np.arctan2(obs['ball'][1], obs['ball'][0])  # Just a placeholder for real angle computation

                    if distance > 0.5:  # Suppose it's a high pass over some threshold distance
                        components['passing_reward'][r_idx] = self.pass_reward
                        reward[r_idx] += components['passing_reward'][r_idx]
                        self.pass_counters[team_idx, zones_covered] += 1

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
