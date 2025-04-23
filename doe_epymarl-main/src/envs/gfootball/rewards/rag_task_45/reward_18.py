import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on abrupt stopping and quick direction changes defensively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._total_stops = 0
        self._stop_reward = 0.05
        self._sprint_stops = {}
    
    def reset(self):
        """Resets environment and internal variables."""
        self._sprint_stops.clear()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the internal state for serialization."""
        state = self.env.get_state(to_pickle)
        state['sprint_stops'] = self._sprint_stops
        return state

    def set_state(self, state):
        """Restores the internal state from serialized data."""
        from_pickle = self.env.set_state(state)
        self._sprint_stops = from_pickle['sprint_stops']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on rewarding rapid changes in movement directions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_sprint_rewards": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for index, obs in enumerate(observation):
            curr_controlled = obs['active']
            if obs is None:
                continue
            
            # Check if player just stopped sprinting
            if self._sprint_stops.get(index, False) and obs['sticky_actions'][8] == 0:
                components["stop_sprint_rewards"][index] = self._stop_reward
                self._sprint_stops[index] = False

            # Reward stopping a sprint
            if obs['sticky_actions'][8] == 1:
                self._sprint_stops[index] = True
            else:
                self._sprint_stops[index] = False
            
            # Updating reward based on stopping
            reward[index] += components["stop_sprint_rewards"][index]

        return reward, components

    def step(self, action):
        """Steps through the environment, wrapping the rewards with additional metrics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
