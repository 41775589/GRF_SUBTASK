import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for clearing the ball from defensive zones under pressure.
    Focuses on whether the clearance was directed away from the goal and into a safe area.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones_threshold = 0.3  # The threshold for what counts as a defensive zone, closer to own goal
        self.clearance_success_reward = 1.0  # Reward given for successful clearance
        self.multiplier_for_pressure = 2.0  # Additional reward multiplier if done under pressure

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
        """
        Custom reward logic to enhance ball clearance under pressure from defensive zones.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            components["clearance_reward"][rew_index] = 0.0

            # Detect if the ball is in a defensive zone
            if o['ball'][0] < -self.defensive_zones_threshold:
                # Check if a clearance action is taking place
                if self.sticky_actions_counter[6] > 0 or self.sticky_actions_counter[0] > 0:  # Bot actions
                    pressure_factor = 1.0
                    if np.abs(o['ball'][1]) < 0.1:  # Ball is centrally located, more risky
                        pressure_factor = self.multiplier_for_pressure

                    components["clearance_reward"][rew_index] = self.clearance_success_reward * pressure_factor
                    reward[rew_index] += components["clearance_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
