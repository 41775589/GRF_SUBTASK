import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper that provides dense rewards based on offensive actions leading to scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_loops = 0
        self.dribble_loops = 0
        self.shot_attempts = 0

    def reset(self):
        """Resets the environment and the counters for action tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_loops = 0
        self.dribble_loops = 0
        self.shot_attempts = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encodes the current state of the wrapper, including specific counters, to be preserved."""
        to_pickle['pass_loops'] = self.pass_loops
        to_pickle['dribble_loops'] = self.dribble_loops
        to_pickle['shot_attempts'] = self.shot_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Decodes and sets the internal state of the wrapper from a pickle."""
        from_pickle = self.env.set_state(state)
        self.pass_loops = from_pickle['pass_loops']
        self.dribble_loops = from_pickle['dribble_loops']
        self.shot_attempts = from_pickle['shot_attempts']
        return from_pickle

    def reward(self, reward):
        """Augments the original reward based on successful offensive actions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": 0.0,
                      "dribble_bonus": 0.0,
                      "shot_bonus": 0.0}

        active_player = observation['active']
        sticky_actions = observation['sticky_actions']

        # Check for successful passes (either long or short)
        if sticky_actions[0] or sticky_actions[1]:  # assuming indices for short and long passes respectively
            self.pass_loops += 1
            components["pass_bonus"] += 0.05 * self.pass_loops

        # Check for dribbles
        if sticky_actions[9]:  # assuming index for dribbling
            self.dribble_loops += 1
            components["dribble_bonus"] += 0.03 * self.dribble_loops

        # Check for shots (assuming index for shot)
        if sticky_actions[2]:
            self.shot_attempts += 1
            components["shot_bonus"] += 0.1 * self.shot_attempts

        # Compute the total modified reward for the active player
        modified_reward = reward[active_player] + sum(components.values())
        reward[active_player] = modified_reward

        return reward, components

    def step(self, action):
        """Takes a step in the environment and augments the reward information returned."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = value
        return observation, reward, done, info
