import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for enhancing close-range defensive skills using 'Sliding'."""

    def __init__(self, env):
        super().__init__(env)
        self._num_sliding_actions = 0
        self._sliding_success_reward = 2.0
        self._proximity_threshold = 0.1  # Proximity to the goal to consider it as 'close-range'
        self._negative_reward_for_miss = -1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int) 

    def reset(self):
        self._num_sliding_actions = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._num_sliding_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_sliding_actions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Access observation to determine if the 'Sliding' action was successful
        observation = self.env.unwrapped.observation()[0]  # Assuming a single agent

        if observation['game_mode'] != 0:  # Only consider rewards if the game is in a normal play mode
            return reward, {"base_score_reward": reward.copy(), "sliding_action_reward": [0.0]}

        components = {"base_score_reward": reward.copy(), "sliding_action_reward": [0.0]}

        # Determine if the agent is close to its own goal and performed a sliding
        is_close_to_goal = np.linalg.norm(observation['ball'][:2]) < self._proximity_threshold
        did_sliding = observation['sticky_actions'][9] == 1  # Assuming 'Sliding' is indexed as 9

        if did_sliding and self._num_sliding_actions == 0:
            if is_close_to_goal:
                # Give a positive reward if the sliding action is performed close to goal
                components["sliding_action_reward"][0] = self._sliding_success_reward
                reward[0] += components["sliding_action_reward"][0]
                self._num_sliding_actions += 1
            else:
                # Negative reward if sliding action is misused
                components["sliding_action_reward"][0] = self._negative_reward_for_miss
                reward[0] += components["sliding_action_reward"][0]

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
