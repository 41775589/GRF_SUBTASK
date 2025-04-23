import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing behaviors typical of wide midfielders, particularly focusing
    on high passes and effective positioning to utilize the width of the pitch.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # tracking sticky actions

        # Reward constants
        self.high_pass_reward = 0.3
        self.width_utilization_reward = 0.5
        # Position thresholds to encourage wide play
        self.lateral_field_threshold = 0.7

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
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for performing a high pass
            if o['sticky_actions'][6] == 1 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["high_pass_reward"][rew_index] += self.high_pass_reward * self.sticky_actions_counter[6]
            
            # Reward for maintaining a position exceeding lateral field threshold
            player_position = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
            if abs(player_position) > self.lateral_field_threshold:
                components["position_reward"][rew_index] += self.width_utilization_reward

            # Update sticky actions count
            self.sticky_actions_counter += o['sticky_actions']

            # Aggregate rewards for each component
            reward[rew_index] += components["high_pass_reward"][rew_index]
            reward[rew_index] += components["position_reward"][rew_index]

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
