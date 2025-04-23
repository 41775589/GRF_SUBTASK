import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a custom reward function focusing on winger performance enhancing.
    This function specifically encourages players to sprint along the wings, perform successful
    dribbles, and deliver accurate crosses.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # tracking sticky actions
        # Constants for tuning the reward function behavior
        self.crossing_reward = 0.5
        self.dribbling_reward = 0.3
        self.sprinting_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_wrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]

            # Encouraging crossing when close to goal area on the wings
            if (abs(o['ball'][0]) > 0.7 and abs(o['ball'][1]) > 0.2) and (o['sticky_actions'][9] == 1):
                components["crossing_reward"][i] = self.crossing_reward

            # Encouraging dribbling when players are active on the field
            if o['sticky_actions'][9] == 1 and (o['ball_owned_team'] == o['side']):
                components["dribbling_reward"][i] += self.dribbling_reward

            # Encouraging sprinting along the wings
            if (abs(o['position'][1]) > 0.2) and (o['sticky_actions'][8] == 1):
                components["sprinting_reward"][i] = self.sprinting_reward

            # Compute the final reward for the player
            reward[i] += components["crossing_reward"][i] + components["dribbling_reward"][i] + components["sprinting_reward"][i]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
