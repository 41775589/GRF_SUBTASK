import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering Wide Midfield responsibilities, focusing on High Pass execution
    and proper positioning to expand the field of play and support lateral transitions.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.05
        self.positioning_reward_multiplier = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Specific states are not being restored as no internal state is stored in this example.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward execution of high pass (index in sticky actions array for high pass action)
            if o['sticky_actions'][5] == 1:  # Assuming index 5 refers to High Pass
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += self.high_pass_reward

            # Positioning reward based on the player's x-position. Encouraging wider positions.
            player_pos = o['left_team'][o['active']]
            positioning_score = abs(player_pos[0])  # Wider is closer to 1 (or -1 for other side)

            # Calculate positioning reward scale based on defending (+) or attacking (-) half
            if player_pos[0] > 0:
                # Reward for being wide in the attacking half
                components["positioning_reward"][rew_index] = self.positioning_reward_multiplier * positioning_score
            else:
                # Lower reward for being wide in the defending half
                components["positioning_reward"][rew_index] = self.positioning_reward_multiplier * (1 - positioning_score)

            reward[rew_index] += components["positioning_reward"][rew_index]

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
                self.sticky_actions_counter[i] = 1 if action > 0 else self.sticky_actions_counter[i]
        return observation, reward, done, info
