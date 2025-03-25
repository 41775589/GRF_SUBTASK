import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on enhancing offensive strategies and precision finishing."""
    def __init__(self, env):
        super().__init__(env)
        self._num_zone_transitions = 5  # Reward attempts to cross these zones toward goal
        self._zone_reward = 0.2  # Reward delivered when agents cross into a new zone toward the opponent's goal
        self._collected_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Initialize a structured reward component dictionary
        components = {'base_score_reward': reward.copy(),
                      'zone_transition_reward': 0.0}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for index in range(len(reward)):
            o = observation[index]
            current_zone = int((o['ball'][0] + 1) * self._num_zone_transitions / 2)  # Normalize and scale ball position

            if self._collected_zones.get(index, -1) < current_zone:
                # Reward for moving the ball into a new offensive zone closer to the goal
                reward[index] += self._zone_reward
                components['zone_transition_reward'] += self._zone_reward
                self._collected_zones[index] = current_zone

            if o['score'][1] > o['score'][0]:  # Check if the right team (controlling team) has scored
                reward[index] += 1  # Reward scoring
                components['score_goal_reward'] = 1.0

        return reward, components
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)  # Record sum of components for multi-agent scenarios
        return observation, reward, done, info
