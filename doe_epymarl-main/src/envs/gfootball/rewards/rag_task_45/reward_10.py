import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward function to train agents on abrupt stopping and direction changes.
    It incentivizes stopping (action 5: action_bottom) and then sprinting (action 8: action_sprint)
    or moving in any direction, reinforcing the Stop-Sprint and Stop-Moving tactics defensively.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and sticky actions counter for the new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Gets the state of the environment, including any necessary wrapper-specific state.
        """
        to_pickle['CheckpointRewardWrapper'] = True  # Example placeholder
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the state of the environment, including any wrapper-specific state extracted from input.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function to enhance the abrupt stopping and quick direction change capabilities of agents.

        Rewarding heavily on successful transition from stopping to sprinting/moving in a quick succession.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        new_reward = []
        for o, base_rew in zip(observation, reward):
            # Initialize component rewards with base reward
            action_reward = base_rew
            
            # Agent stops
            if o['sticky_actions'][5] == 1:
                action_reward += 0.5  # Reward for stopping

            # After stopping, agent sprints or moves in any direction
            if sum(o['sticky_actions'][:4]) > 0 or o['sticky_actions'][8] == 1:
                action_reward += 2.0  # Reward for sprinting or moving directly after stopping
            
            new_reward.append(action_reward)
        
        # Save the new rewards along with their components for analysis
        return new_reward, components

    def step(self, action):
        """
        Takes an action in the environment, applies the custom reward function, and returns the results.

        Returns:
            observation, reward, done, info
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
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
