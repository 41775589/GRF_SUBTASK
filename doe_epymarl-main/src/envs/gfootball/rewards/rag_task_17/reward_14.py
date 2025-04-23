import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for effectively managing wide midfield responsibilities.
    Rewards are granted for high passes and effective positioning at the sides of the field to support lateral transitions
    and stretch the opposition's defense.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state_info = self.env.set_state(from_pickle)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return state_info

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for high passes, observed by sticky_actions[7]
            if o['sticky_actions'][7] == 1:  # Assuming index 7 corresponds to the high pass action
                components["high_pass_reward"][rew_index] = 0.2
                self.sticky_actions_counter[7] += 1
            
            # Reward for effective lateral positioning
            # Check if the player is near the sides relative to their team's field section
            player_x, player_y = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            if abs(player_y) > 0.25:  # A threshold for deciding that the player is effectively wide
                components["positioning_reward"][rew_index] = 0.1

            # Summing up all rewards
            reward[rew_index] += components["high_pass_reward"][rew_index] + components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for ai, agent_obs in enumerate(obs):
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_state
                info[f"sticky_actions_{i}_{ai}"] = action_state
        return observation, reward, done, info
