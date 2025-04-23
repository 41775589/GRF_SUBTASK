import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for executing accurate long passes in specific areas,
    focusing on vision, timing, and precision in ball distribution.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._long_pass_zones = [(0.5, 0), (-0.5, 0)]  # Target zones for long passes
        self._pass_accuracy_reward = 1.0  # Reward for accurate pass in target zone

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
        Augment the player reward based on successful long passes to defined zones.
        """
        observation = self.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        for idx, obs in enumerate(observation):
            # Check if a long pass has occurred
            if obs['ball_direction'][0] ** 2 + obs['ball_direction'][1] ** 2 > 0.3 ** 2:  # Arbitrary threshold for "long pass"
                final_ball_pos_x = obs['ball'][0] + obs['ball_direction'][0]
                final_ball_pos_y = obs['ball'][1] + obs['ball_direction'][1]
                # Check if the ball ends up in either of the target zones
                for zone in self._long_pass_zones:
                    if np.sqrt((final_ball_pos_x - zone[0]) ** 2 + (final_ball_pos_y - zone[1]) ** 2) < 0.2:  # Zone radius threshold
                        components["long_pass_reward"][idx] = self._pass_accuracy_reward
                        reward[idx] += self._pass_accuracy_reward
                        break

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return obs, reward, done, info
