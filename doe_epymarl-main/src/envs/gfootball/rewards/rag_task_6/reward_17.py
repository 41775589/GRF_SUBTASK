import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds stamina and movement strategy awareness through dense rewards,
    focusing on efficient Stop-Sprint and Stop-Moving actions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define a reward for efficient energy usage
        self.sprint_stop_efficiency_reward = 0.05
        self.movement_stop_efficiency_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        current_observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_stop_reward": [0.0] * len(reward),
                      "movement_stop_reward": [0.0] * len(reward)}

        if current_observation is None:
            return reward, components

        for i in range(len(current_observation)):
            player_obs = current_observation[i]

            # Sprint control: reward for stopping sprint efficiently
            if player_obs['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] > 0:
                components["sprint_stop_reward"][i] = self.sprint_stop_efficiency_reward
                reward[i] += components["sprint_stop_reward"][i]

            # General movement control: reward for stopping movement efficiently
            if (sum(player_obs['sticky_actions'][:8]) == 0 and 
                sum(self.sticky_actions_counter[:8]) > 0):
                components["movement_stop_reward"][i] = self.movement_stop_efficiency_reward
                reward[i] += components["movement_stop_reward"][i]
            
            # Update counters for sticky actions
            self.sticky_actions_counter = player_obs['sticky_actions']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Continuous update of the sticky actions state
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
