import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on training goalkeeper skills in shot stopping,
    decision-making under pressure, and effective communication with defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.save_attempts = 0
        self.save_rewards = 0.5
        self.ball_distribution_rewards = 0.3

    def reset(self):
        """Resets the environment and rewards metrics."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.save_attempts = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Returns the state of the environment for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the environment state from serialized data."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function designed for goalkeeper training."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0] * len(reward),
                      "distribution_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward goalkeeper for successful saves
            if o['game_mode'] in [6]:  # Penalty or similar high-pressure modes
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    components["save_reward"][rew_index] = self.save_rewards
                    self.save_attempts += 1

            # Reward distribution effectiveness under pressure
            if o['ball_owned_team'] == 0 and o['active'] == o['designated']:
                if np.any(o['sticky_actions'][7:10]):  # check if bottom actions are used to distribute ball
                    components["distribution_reward"][rew_index] = self.ball_distribution_rewards

            reward[rew_index] += components["save_reward"][rew_index] + components["distribution_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
