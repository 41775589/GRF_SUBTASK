import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards based on the effective use of Sprint and Stop-Sprint actions,
    with an emphasis on optimizing defensive positions using speed adjustments.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the sticky actions counter on environment reset.
        """
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Gets the current state of the reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the state of the reward wrapper from saved state values.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Adapts rewards based on the strategic use of sprint and stop-sprint actions to quickly adjust positions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positioning_reward": [0.0]}

        sprint_index, stop_sprint_index = 8, 9
        sprint_active = observation['sticky_actions'][sprint_index] == 1
        stop_sprint_active = observation['sticky_actions'][stop_sprint_index] == 1

        # Parameters for reward adjustment
        reward_sprint = 0.1  # Sprint reward
        reward_stop_sprint = 0.05  # Stop sprint reward

        # Reward for sprinting: Encourage if it results in significant position change
        if sprint_active:
            self.sticky_actions_counter[sprint_index] += 1
            current_position = observation.get('right_team')[observation['active']]
            previous_position = observation.get('right_team')[observation['active'] - 1]
            distance_moved = distance.euclidean(current_position, previous_position)

            if distance_moved > 0.05:  # Threshold for "significant" movement
                components["positioning_reward"][0] += reward_sprint

        # Reward for stopping sprint: Encourage if it helps in quick position stabilization
        if stop_sprint_active:
            self.sticky_actions_counter[stop_sprint_index] += 1
            components["positioning_reward"][0] += reward_stop_sprint

        # Update the total reward with the additional component for dynamic position adjustment
        reward[0] += components["positioning_reward"][0]
        
        return reward, components

    def step(self, action):
        """
        Steps through the environment, applying action and altering reward based on defined criteria.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions count for information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for i, action in enumerate(obs['sticky_actions']):
            info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
