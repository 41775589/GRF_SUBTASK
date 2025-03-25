import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering high passes and wide midfielder positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.2
        self.positioning_reward = 0.1
        self._num_high_passes = 0
        self._wide_positions_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_high_passes = 0
        self._wide_positions_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'high_passes': self._num_high_passes, 
            'positions': self._wide_positions_collected
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_high_passes = from_pickle['CheckpointRewardWrapper']['high_passes']
        self._wide_positions_collected = from_pickle['CheckpointRewardWrapper']['positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        final_rewards = reward.copy()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            # Reward for successfully performing a high pass
            if 'sticky_actions' in o:
                high_pass = (o['sticky_actions'][8] == 1)  # Index for high pass action in sticky actions
                if high_pass:
                    self._num_high_passes += 1
                    final_rewards[rew_index] += self.high_pass_reward
                    components["high_pass_reward"][rew_index] = self.high_pass_reward

            # Reward for being in a wide midfielder lateral position
            x, y = o['ball']
            if -1 <= x <= -0.5 or 0.5 <= x <= 1:  # Wide areas of the field
                pos_key = (rew_index, round(x, 1), round(y, 1))
                if pos_key not in self._wide_positions_collected:
                    self._wide_positions_collected[pos_key] = True
                    final_rewards[rew_index] += self.positioning_reward
                    components["position_reward"][rew_index] = self.positioning_reward

        return final_rewards, components

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
