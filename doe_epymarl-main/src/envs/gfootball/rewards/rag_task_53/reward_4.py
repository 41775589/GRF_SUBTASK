import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on ball control and strategic play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion = {}
        self.control_duration = {}
        self.spatial_coverage = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion = {}
        self.control_duration = {}
        self.spatial_coverage = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['pass_completion'] = self.pass_completion
        state['control_duration'] = self.control_duration
        state['spatial_coverage'] = list(self.spatial_coverage)
        return state

    def set_state(self, state):
        self.pass_completion = state['pass_completion']
        self.control_duration = state['control_duration']
        self.spatial_coverage = set(state['spatial_coverage'])
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "control_duration_reward": [0.0] * len(reward),
                      "spatial_coverage_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for pass completion to a teammate
            if o.get('ball_owned_team') == o['active'] and o['active'] not in self.pass_completion:
                components["pass_completion_reward"][rew_index] = 0.5
                self.pass_completion[o['active']] = True
            reward[rew_index] += components["pass_completion_reward"][rew_index]

            # Reward for duration of controlling the ball
            if o.get('ball_owned_team') == o['active']:
                self.control_duration[rew_index] = self.control_duration.get(rew_index, 0) + 1
            if self.control_duration.get(rew_index, 0) > 50:  # arbitrary duration for holding control
                components["control_duration_reward"][rew_index] = 0.1
            reward[rew_index] += components["control_duration_reward"][rew_index]

            # Reward for covering different grid sections on the pitch
            grid_position = (int(o['ball'][0] * 5), int(o['ball'][1] * 5))  # 5x5 grid approximation
            if grid_position not in self.spatial_coverage:
                self.spatial_coverage.add(grid_position)
                components["spatial_coverage_reward"][rew_index] = 0.2
            reward[rew_index] += components["spatial_coverage_reward"][rew_index]

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
