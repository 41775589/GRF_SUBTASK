import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward to encourage progressive ball control and evading capabilities.
    This reward structure promotes quick sprints and efficient dribbling through defensive lines.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._visited_regions = {}
        self._num_regions = 5
        self._region_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._visited_regions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._visited_regions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._visited_regions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == 1:
                components["checkpoint_reward"][rew_index] = self._region_reward * (
                        self._num_regions -
                        self._visited_regions.get(rew_index, 0))
                reward[rew_index] = components["base_score_reward"][rew_index] + components["checkpoint_reward"][rew_index]
                self._visited_regions[rew_index] = self._num_regions
                continue

            if o['ball_owned_team'] == 1 and 'sticky_actions' in o and o['sticky_actions'][8]: # Checks if sprint action is active
                x_position = o['ball'][0] # X position of the ball
                region_index = int(min(x_position * (self._num_regions / 2) + (self._num_regions / 2), self._num_regions - 1))

                if region_index > self._visited_regions.get(rew_index, 0):
                    components["checkpoint_reward"][rew_index] = self._region_reward
                    reward[rew_index] += components["checkpoint_reward"][rew_index]
                    self._visited_regions[rew_index] = region_index

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
