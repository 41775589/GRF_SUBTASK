import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long-distance shots from outside the penalty box."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.distance_threshold = 0.6  # Roughly outside the penalty box distance
        self.long_shot_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        self.env.set_state(from_pickle)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_shot_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball_owned_team'] == 1 and o['ball'][0] > self.distance_threshold:
                # Check if the ball is far from the goal and the right team has possession
                if o['score'][1] > o['score'][0]:  # Right team scored
                    components["long_shot_reward"][rew_index] += self.long_shot_reward
                    reward[rew_index] += self.long_shot_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
