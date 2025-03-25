import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on long pass accuracy and dynamics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_origin = None
        self.long_pass_threshold = 0.3  # Threshold to determine a long pass based on distance
        self.long_pass_reward = 0.5  # Extra reward for successful long pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_origin = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'pass_origin': self.pass_origin
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_origin = from_pickle['CheckpointRewardWrapper']['pass_origin']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' not in o or o['ball_owned_team'] != o['active']:
                self.pass_origin = None
                continue

            if self.pass_origin is not None:
                # Calculate the distance the ball has travelled since the long pass initiation
                current_pos = o['ball'][:2]
                distance = np.linalg.norm(np.array(self.pass_origin) - np.array(current_pos))
                if distance > self.long_pass_threshold:
                    components["long_pass_reward"][rew_index] = self.long_pass_reward
                    reward[rew_index] += components["long_pass_reward"][rew_index]
                    self.pass_origin = None  # Reset after rewarding to prevent multiple rewards for the same pass

            # Check if a new pass is initiated
            if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0:
                self.pass_origin = o['ball'][:2]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
