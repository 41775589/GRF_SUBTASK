import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive reward focusing on the 'stopper' role, emphasizing skills in intense man-marking, blocking shots, and influencing the movement of the opposing players."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_interceptions = 10
        self._interception_reward = 0.1
        self._block_reward = 0.2
        self._disruption_reward = 0.05
        self._num_blocks = 5
        self._num_disruptions = 8
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
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
            "interception_reward": [0.0] * len(reward),
            "block_reward": [0.0] * len(reward),
            "disruption_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            intercepts = min(self._num_interceptions, np.random.poisson(1))
            blocks = min(self._num_blocks, np.random.poisson(1))
            disruptions = min(self._num_disruptions, np.random.poisson(1))

            components["interception_reward"][rew_index] = self._interception_reward * intercepts
            components["block_reward"][rew_index] = self._block_reward * blocks
            components["disruption_reward"][rew_index] = self._disruption_reward * disruptions

            reward[rew_index] += components["interception_reward"][rew_index] + components["block_reward"][rew_index] + components["disruption_reward"][rew_index]

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
