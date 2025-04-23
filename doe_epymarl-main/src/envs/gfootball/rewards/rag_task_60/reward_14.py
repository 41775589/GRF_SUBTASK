import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for transitions between movement and stopping to enhance defensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for defensive stops
        self._stop_window = 10  # Number of steps to consider for stopping
        self._moving_threshold = 0.02  # Minimum movement threshold to consider as moving
        self._stop_reward = 0.2  # Reward for successful stop after movement
        self._movement_history = []
        self._last_positions = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._movement_history = []
        self._last_positions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'movement_history': self._movement_history,
            'last_positions': self._last_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._movement_history = from_pickle['CheckpointRewardWrapper']['movement_history']
        self._last_positions = from_pickle['CheckpointRewardWrapper']['last_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            current_positions = observation[rew_index]['left_team'] + observation[rew_index]['right_team']
            if self._last_positions is not None:
                movements = np.linalg.norm(current_positions - self._last_positions, axis=1)
                was_moving = [m > self._moving_threshold for m in movements]
                self._movement_history.append(was_moving)

            if len(self._movement_history) >= self._stop_window:
                recent_movements = self._movement_history[-self._stop_window:]
                transition_occurred = [all(m) and not recent_movements[-1][i] for i, m in enumerate(zip(*recent_movements))]
                reward[rew_index] += self._stop_reward * sum(transition_occurred)  # Reward for each stopping transition
                components["defensive_transition_reward"][rew_index] = self._stop_reward * sum(transition_occurred)

            self._last_positions = current_positions
        
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
