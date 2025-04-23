import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for precise control using Stop-Dribble under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Adjust these limits as per the observation specifics and game requirements
        self.dribble_intensity_threshold = 0.5
        self.pressure_threshold = 0.3  # Hypothetical measure of closeness of opponents

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load any desired state here if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_control_reward": [0.0, 0.0]
        }

        if observation is None:
            return reward, components
        
        for agent_idx in range(len(reward)):
            o = observation[agent_idx]
            if o['sticky_actions'][9] == 1 and self.evaluate_pressure(o) > self.pressure_threshold:
                # Check dribble action is active under pressure
                components["dribble_control_reward"][agent_idx] = self.calculate_dribble_reward(o)
                reward[agent_idx] += components["dribble_control_reward"][agent_idx]
        
        return reward, components

    def evaluate_pressure(self, observation):
        # Simplified example, ideally use the positions of opponents relative to player
        return np.random.uniform(0, 1)  # Placeholder implementation

    def calculate_dribble_reward(self, observation):
        # Assess how effectively the dribble was stopped under pressure
        dribble_quality = np.random.uniform(0, 1)  # Placeholder implementation
        return dribble_quality if dribble_quality > self.dribble_intensity_threshold else 0

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
