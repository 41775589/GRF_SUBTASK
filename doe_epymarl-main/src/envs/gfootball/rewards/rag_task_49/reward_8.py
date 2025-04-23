import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for shooting accuracy and power from the central field."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._shooting_zones_rewards = [0.5, 0.75, 1.0]  # for mid-field and near-goal with more for direct goal zone
        self._accuracy_reward_coefficient = 0.2

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
        components = {"base_score_reward": reward.copy(), "accuracy_power_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o.get('ball', [0, 0, 0])
            x_pos = ball_position[0]

            # Reward for attempts made at specific zones (central field to opponent's goal)
            if np.abs(x_pos) < 0.2:  # Central field
                components["accuracy_power_reward"][rew_index] = self._shooting_zones_rewards[0]
            elif np.abs(x_pos) < 0.5:  # Approaching goal
                components["accuracy_power_reward"][rew_index] = self._shooting_zones_rewards[1]
            elif np.abs(x_pos) >= 0.75 and x_pos > 0:  # Very close to goal
                components["accuracy_power_reward"][rew_index] = self._shooting_zones_rewards[2]

            # Additional accuracy bonus for how central the shot is towards the goal (using y coordinate)
            y_pos = ball_position[1]
            components["accuracy_power_reward"][rew_index] += self._accuracy_reward_coefficient * (1 - np.abs(y_pos))

            reward[rew_index] += components["base_score_reward"][rew_index] + components["accuracy_power_reward"][rew_index]

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
