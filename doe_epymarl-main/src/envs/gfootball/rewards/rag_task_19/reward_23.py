import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic midfield control and defensive maneuvers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_control_weight = 0.2
        self.defensive_play_weight = 0.3
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self.previous_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "defensive_action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward midfield control based on ball position
            if o['ball'][0] > -0.3 and o['ball'][0] < 0.3:
                components["midfield_control_reward"][rew_index] = self.midfield_control_weight
                reward[rew_index] += components["midfield_control_reward"][rew_index]

            # Reward defensive actions based on ball proximity to goal
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball']) > 0.6:
                components["defensive_action_reward"][rew_index] = self.defensive_play_weight
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
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        return observation, reward, done, info
