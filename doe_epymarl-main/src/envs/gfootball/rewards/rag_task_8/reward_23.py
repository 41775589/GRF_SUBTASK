import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes quick decision-making and efficient ball handling.
    It adds a reward for immediately initiating a counter-attack after recovering possession.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.changing_possession_reward = 0.2  # Reward gain for changing possession and performing quick action.
        self.last_possession = None  # Keep track of last team possession
        self.reset_counter = 5  # Count after possession change to provide reward once

    def reset(self):
        """
        Reset the environment and the internal state for the reward wrapper.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_possession = None
        self.reset_counter = 5
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Get the state of the environment and the reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {'last_possession': self.last_possession,
                                                 'reset_counter': self.reset_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment and the reward wrapper from a pickle object.
        """
        from_pickle = self.env.set_state(state)
        self.last_possession = from_pickle['CheckpointRewardWrapper']['last_possession']
        self.reset_counter = from_pickle['CheckpointRewardWrapper']['reset_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward given by the environment.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "changing_possession_reward": 0.0}

        # Check ball possession change
        current_possession = observation['ball_owned_team']
        if current_possession != self.last_possession and current_possession != -1:
            if self.reset_counter > 0:
                # Reward is granted if the counter-attack is initiated quickly
                components["changing_possession_reward"] = self.changing_possession_reward
                reward += components["changing_possession_reward"]
                self.reset_counter -= 1
        else:
            self.reset_counter = 5

        self.last_possession = current_possession
        return reward, components

    def step(self, action):
        """
        Execute a step in the environment with additional wrapped reward calculation.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
