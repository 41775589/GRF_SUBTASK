import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on high pass effectivity and granting possession in wide midfield positions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.wide_midfield_checkpoints = {}
        self.high_pass_reward = 0.05
        self.position_reward = 0.03
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.wide_midfield_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.wide_midfield_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.wide_midfield_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            player_x = o['right_team'][:, 0] if o['ball_owned_team'] == 1 else o['left_team'][:, 0]
            player_y = o['right_team'][:, 1] if o['ball_owned_team'] == 1 else o['left_team'][:, 1]

            active = o['active']
            action_set = np.array(o['sticky_actions'])

            # Check for high pass (index 4 for a high pass action)
            if action_set[4] == 1:
                components['high_pass_reward'][rew_index] = self.high_pass_reward
                reward[rew_index] += components['high_pass_reward'][rew_index]

            # Reward for wide midfield positioning
            if -0.3 <= player_y[active] <= 0.3 and (player_x[active] < -0.6 or player_x[active] > 0.6):
                components['position_reward'][rew_index] = self.position_reward
                reward[rew_index] += components['position_reward'][rew_index]

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
