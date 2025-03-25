import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that implements a reward function focused on the role of a 'sweeper.'
    Rewards are designed to encourage clearing the ball from the defensive zone,
    performing critical last-man tackles, and supporting by covering positions and executing fast recoveries.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.3
        self.last_man_tackle_reward = 0.5
        self.cover_position_reward = 0.2
        self.fast_recovery_reward = 0.4

    def reset(self):
        """
        Resets the environment and any wrapper-specific data.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Encapsulates the wrapper-specific state in a pickle-able object.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restores the wrapper-specific state from the given state object.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward returned by the environment by adding specific tasks achievements:
        - Reward for clearing the ball out of the defensive zone.
        - Reward for successful last-man tackles.
        - Reward for fast recovery to a position and covering gaps when a teammate advances or tackles.

        Parameters:
            reward (float): The initial reward to process.

        Returns:
            float: The modified reward including additional rewards for performed tasks.
        """
        observation = self.env.unwrapped.observation()
        # Decompose the components for detailed reward manipulation
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward),
            "last_man_tackle_reward": [0.0] * len(reward),
            "cover_position_reward": [0.0] * len(reward),
            "fast_recovery_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward

        for i, o in enumerate(observation):
            # Assumption: 'is_last_man_defense' and other flags are set based on the game state analysis
            if o.get('is_clearing_ball', False):
                components["clearance_reward"][i] = self.clearance_reward
            if o.get('is_last_man_tackle', False):
                components["last_man_tackle_reward"][i] = self.last_man_tackle_reward
            if o.get('is_covering_position', False):
                components["cover_position_reward"][i] = self.cover_position_reward
            if o.get('is_fast_recovery', False):
                components["fast_recovery_reward"][i] = self.fast_recovery_reward
            
            # Apply additional rewards to the base reward
            reward[i] += (components["clearance_reward"][i] + 
                         components["last_man_tackle_reward"][i] +
                         components["cover_position_reward"][i] +
                         components["fast_recovery_reward"][i])

        return reward, components

    def step(self, action):
        """
        Executes an environment step while capturing detailed reward components and modifying the base rewards.

        Returns:
            observation, reward, done, info
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Append reward components to info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        # Reset sticky action counts
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
