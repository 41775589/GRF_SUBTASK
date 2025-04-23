import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds defensive adaptation rewards based on stopping and starting movements."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.balance_threshold = 1  # Threshold for considering defensive movements
        self.movement_stop_reward = 0.1  # Reward for successful stopping at a crucial moment
        self.movement_start_reward = 0.05  # Reward for starting movement in reaction to attack
        self.required_defensive_stops = 5 
        self.stop_counter = np.zeros(2, dtype=int)
        self.start_counter = np.zeros(2, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_counter.fill(0)
        self.start_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_defense_wrapper'] = {
            'stop_counter': self.stop_counter.copy(),
            'start_counter': self.start_counter.copy()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.stop_counter = from_pickle['checkpoint_defense_wrapper']['stop_counter']
        self.start_counter = from_pickle['checkpoint_defense_wrapper']['start_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_reward": [0.0] * len(reward),
                      "start_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Checking defensive stops
            if np.linalg.norm(o['ball_direction']) < self.balance_threshold:
                if self.stop_counter[i] < self.required_defensive_stops:
                    self.stop_counter[i] += 1
                    components["stop_reward"][i] = self.movement_stop_reward
                    reward[i] += components["stop_reward"][i]

            # Checking start of movement
            if np.linalg.norm(o['ball_direction']) > self.balance_threshold:
                if self.start_counter[i] < self.required_defensive_stops:
                    self.start_counter[i] += 1
                    components["start_reward"][i] = self.movement_start_reward
                    reward[i] += components["start_reward"][i]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
