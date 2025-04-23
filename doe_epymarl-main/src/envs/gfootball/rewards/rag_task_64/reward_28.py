import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes and crosses in dynamic attacking scenarios."""
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_completion_reward = 0.5
        self.cross_completion_to_key_area_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset the environment and the sticky actions counter at the start of an episode. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the state of the environment to be saved or pickled. """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state of the environment from the saved or unpickled state. """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """ Modify reward based on special game events related to high passes and crosses. """
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0],
                      "cross_completion_reward": [0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward for successful high passes.
        high_pass_action_set = {7, 8}  # Assuming these indices correspond to high pass actions.
        for rew_index in range(len(reward)):
            if observation[rew_index]['sticky_actions'][7] == 1 or observation[rew_index]['sticky_actions'][8] == 1:
                components["pass_completion_reward"][rew_index] += self.pass_completion_reward
                reward[rew_index] += components["pass_completion_reward"][rew_index]

        # Reward for successful crosses into a key attacking area.
        if 'ball' in observation[0] and observation[0]['ball'][0] > 0.5:  # ball in opponent's half
            if observation[0]['ball'][1] > 0.3 or observation[0]['ball'][1] < -0.3:  # ball in wide areas
                components["cross_completion_reward"][0] += self.cross_completion_to_key_area_reward
                reward[0] += components["cross_completion_reward"][0]

        return reward, components

    def step(self, action):
        """ Take a step in the environment using the action and modify the reward accordingly. """
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
