import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that improves the reward for effectively using 'Sliding' to block close-range shots or dribbles near 
    the goal, enhancing defensive skills.
    """

    def __init__(self, env):
        super().__init__(env)
        self._sliding_success_reward = 3.0  # Increased magnitude for substantial influence
        self._proximity_threshold = 0.1  # Defines close range near the goal
        self._miss_penalty = -1.0  # Penalty for sliding misapplication
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sliding_action_index = 9  # Index for 'Sliding' in sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Assuming a single agent and based on proximity to the own goal to apply sliding action logic
        components = {"base_score_reward": reward.copy(), "sliding_action_reward": [0.0]}

        player_pos = observation[0]['left_team'][observation[0]['active']]
        goal_pos = [-1, 0]  # Assuming that the own goal is at left of the field
        distance_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

        if distance_to_goal < self._proximity_threshold:
            sliding_action = observation[0]['sticky_actions'][self._sliding_action_index] == 1

            if sliding_action:
                # Provide a positive reward for successful defense near goal
                components["sliding_action_reward"][0] += self._sliding_success_reward
                reward[0] += components["sliding_action_reward"][0]
            else:
                # Apply penalty if not using sliding near goal
                components["sliding_action_reward"][0] += self._miss_penalty
                reward[0] += components["sliding_action_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Include the sum of rewards in the info for debugging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Keep track of actions used during the step
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
