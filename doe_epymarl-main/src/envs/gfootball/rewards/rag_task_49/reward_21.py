import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a shooting accuracy and power reward from central field positions.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.centers = [(0, y) for y in np.linspace(-0.42, 0.42, num=5)]  # Centric positions on the field
        self.threshold_distance = 0.1  # Distance threshold to consider close to a center
        self.reward_for_shooting = 0.5
        self.power_threshold = 0.7  # Threshold for considering a high power kick

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "accuracy_power_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if shooting action is taken
            if o['sticky_actions'][9]:  # Assuming index 9 corresponds to the shoot action
                ball_pos = o['ball'][:2]  # Get x, y coordinates
                action_power = np.linalg.norm(o['ball_direction'][:2])

                # Check if the player is in a centric zone
                is_central = any(np.linalg.norm(np.array(center) - np.array(ball_pos)) <= self.threshold_distance
                                 for center in self.centers)

                if is_central and action_power > self.power_threshold:
                    components["accuracy_power_reward"][rew_index] += self.reward_for_shooting
                    reward[rew_index] += components["accuracy_power_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
