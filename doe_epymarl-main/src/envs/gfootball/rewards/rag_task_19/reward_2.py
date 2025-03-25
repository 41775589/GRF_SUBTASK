import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing training for defensive maneuvers in midfield management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_rewards = {}
        self.defensive_rewards = {}
        self.defense_midfield_link = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_rewards = {}
        self.defensive_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control_rewards'] = self.midfield_control_rewards
        to_pickle['defensive_rewards'] = self.defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_rewards = from_pickle['midfield_control_rewards']
        self.defensive_rewards = from_pickle['defensive_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward),
            "midfield_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]

            # Defense reward based on ball clearance from defensive third
            if o['ball'][0] < -0.5 and o['left_team'][0][0] < -0.5:
                components["defensive_rewards"][i] += 0.02

            # Midfield control reward based on maintaining possession in the midfield area
            if -0.25 <= o['ball'][0] <= 0.25:
                components["midfield_rewards"][i] += 0.03

            # Linking defensive actions and midfield control
            reward[i] += components["defensive_rewards"][i] + components["midfield_rewards"][i]
            reward[i] += self.defense_midfield_link * (components["defensive_rewards"][i] * components["midfield_rewards"][i])

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
