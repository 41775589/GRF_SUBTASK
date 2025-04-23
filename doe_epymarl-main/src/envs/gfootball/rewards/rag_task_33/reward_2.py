import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for practicing shots from long distance."""

    def __init__(self, env):
        super().__init__(env)
        self.distance_threshold = 0.5  # Considered 'long distance'
        self.shot_power_threshold = 0.7  # Minimum power to consider it a 'shot'
        self.long_shot_reward = 1.0  # Reward for taking long shots

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            if reward[rew_index] != 0:  # Process only on score changes
                continue
            o = observation[rew_index]
            # Check if the shot was made from a long distance
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball']) > self.distance_threshold:
                # Additionally check if the ball is moving towards the goal at high speed
                shoot_direction = np.dot(o['ball_direction'], np.array([-1, 0, 0]))
                if shoot_direction < 0 and np.linalg.norm(o['ball_direction']) > self.shot_power_threshold:
                    components["long_shot_reward"][rew_index] = self.long_shot_reward
                    reward[rew_index] += components["long_shot_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
