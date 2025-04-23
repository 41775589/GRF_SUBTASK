import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments reward based on dribbling skills close to the goalkeeper."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["dribbling_reward"][rew_index] = 0.0

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # our team has the ball
                player_pos = o['left_team'][o['active']]
                goalkeeper_pos = o['right_team'][0]  # assuming index 0 is the goalkeeper

                distance_to_goalkeeper = np.linalg.norm(player_pos - goalkeeper_pos)
                dribbling = o['sticky_actions'][9]  # index 9 corresponds to dribbling action

                # Encourage dribbling when near the goalkeeper
                if dribbling and distance_to_goalkeeper < 0.1:
                    components["dribbling_reward"][rew_index] = 0.5

            # Update cumulative reward
            reward[rew_index] += components["dribbling_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
