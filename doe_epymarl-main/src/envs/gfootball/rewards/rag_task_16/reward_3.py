import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on precise high passing skills in scenarios
    where high passes are advantageous.
    """
    def __init__(self, env):
        super().__init__(env)
        self.ball_height_threshold = 0.15  # Threshold for considering a pass as "high"
        self.precision_threshold = 0.05  # Max distance for "precise" to target position
        self.power_reward_coefficient = 1.0
        self.precision_reward_coefficient = 1.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),
                      "power_reward": [0.0] * len(reward),
                      "precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            if o['ball'][2] > self.ball_height_threshold:  # Ball z position for height of pass
                # Calculate the power component based on the ball's z-direction speed
                if abs(o['ball_direction'][2]) >= self.ball_height_threshold:
                    components["power_reward"][i] = self.power_reward_coefficient

                # Calculate precision component assuming target y position is within game field boundaries
                if abs(o['ball'][0]) < 1 and abs(o['ball'][1]) < 0.42:
                    distance_to_target = np.linalg.norm(o['ball'][:2])
                    if distance_to_target < self.precision_threshold:
                        components["precision_reward"][i] = self.precision_reward_coefficient
            
                # Update the total reward
                reward[i] += components["power_reward"][i] + components["precision_reward"][i]

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
