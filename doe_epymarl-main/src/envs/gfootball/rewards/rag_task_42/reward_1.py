import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamics mastering reward based on ball control and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control = 0.2  # Additional reward for controlling the ball in the midfield

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['MidfieldDynamics'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['MidfieldDynamics']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_dynamics_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            ball_x = obs['ball'][0]

            # If ball is in midfield (-0.2 < x < 0.2) and our team has control
            if -0.2 <= ball_x <= 0.2 and obs['ball_owned_team'] == 0:
                components["midfield_dynamics_reward"][i] = self.midfield_control
                reward[i] += components["midfield_dynamics_reward"][i]

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
