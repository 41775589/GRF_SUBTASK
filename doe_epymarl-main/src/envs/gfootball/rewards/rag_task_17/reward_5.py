import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards players for effective use of the wide areas of the pitch,
    focusing on high passes and maintaining a position that stretches the opponent's defense.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.1  # Reward for successful high passes
        self.positional_reward = 0.05  # Reward for maintaining correct positioning

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positional_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # reward for using high pass
            if o['sticky_actions'][5] == 1:  # Assuming index 5 corresponds to high pass
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # reward for position on the wide areas of the pitch (.42 is the boundary for wide areas)
            player_y_position = o['left_team'][o['active']][1] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])][1]
            if abs(player_y_position) > 0.42:
                components["positional_reward"][rew_index] = self.positional_reward
                reward[rew_index] += components["positional_reward"][rew_index]

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
