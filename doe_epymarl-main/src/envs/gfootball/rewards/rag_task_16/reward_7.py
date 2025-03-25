import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for executing high passes with precision.
    Rewards are based on controlling the trajectory and timing of the pass.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_threshold = 0.8  # Define a quality threshold for a good high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = dict()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, r):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": r.copy(),
                      "high_pass_reward": [0.0] * len(r)}

        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and o['sticky_actions'][9] > 0:
                ball_trajectory = np.linalg.norm(o['ball_direction'])
                if ball_trajectory > self.pass_quality_threshold:
                    components["high_pass_reward"][i] = 0.1  # Give reward for successful high pass
                    r[i] += components["high_pass_reward"][i]

        return r, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
