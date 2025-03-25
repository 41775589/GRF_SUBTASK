import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for midfield control and defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_reward = 0.05
        self.defensive_action_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "defensive_action_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if 'left_team' in o else o['right_team'][o['active']]
            ball_pos = o['ball']

            # Encourage controlling the midfield area
            if abs(player_pos[0]) < 0.25:  # midfield region in [-0.25, 0.25] x-axis
                components["midfield_control_reward"][rew_index] = self.midfield_control_reward
                reward[rew_index] += components["midfield_control_reward"][rew_index]

            # Reward defensive actions when opposition is near team's goal area
            if 'left_team' in o and player_pos[0] < -0.7 and ball_pos[0] < -0.7:
                components["defensive_action_reward"][rew_index] = self.defensive_action_reward
                reward[rew_index] += components["defensive_action_reward"][rew_index]
            elif 'right_team' in o and player_pos[0] > 0.7 and ball_pos[0] > 0.7:
                components["defensive_action_reward"][rew_index] = self.defensive_action_reward
                reward[rew_index] += components["defensive_action_reward"][rew_index]

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
