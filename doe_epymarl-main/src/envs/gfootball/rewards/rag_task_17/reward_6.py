import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering wide midfield responsibilities,
    such as high pass usage and effective positioning to stretch the opposition's defense.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.effective_pass_threshold = 0.5  # Threshold to consider a pass effective based on position change
        self.high_pass_reward_coefficient = 0.1
        self.positioning_reward_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_effectiveness_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Determining high pass effectiveness
            if o['sticky_actions'][9] and o['ball_owned_team'] in [0, 1]:
                ball_start_pos = o['ball']
                ball_end_pos = self.env.unwrapped.next_state['ball']
                position_change = np.linalg.norm(ball_end_pos[:2] - ball_start_pos[:2])

                if position_change > self.effective_pass_threshold:
                    components["high_pass_effectiveness_reward"][rew_index] = self.high_pass_reward_coefficient
                    reward[rew_index] += self.high_pass_reward_coefficient

            # Rewarding for effective wide midfield positioning
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            if abs(player_pos[1]) > 0.3:  # Checking if the player is close to the sidelines
                components["positioning_reward"][rew_index] = self.positioning_reward_coefficient
                reward[rew_index] += self.positioning_reward_coefficient

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
