import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for promoting team synergy during possession changes."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize the sticky action counter for players.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Set the reward values for our custom rewards.
        self.possession_change_reward = 0.1
        # A dictionary to store information regarding possession per episode.
        self.possession_states = {}

    def reset(self):
        """Resets the environment state at the beginning of a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_states = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encodes the current state to save with possession states."""
        to_pickle['CheckpointRewardWrapperPossessionStates'] = self.possession_states
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores from the saved state."""
        from_pickle = self.env.set_state(state)
        self.possession_states = from_pickle.get('CheckpointRewardWrapperPossessionStates', {})
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on team synergy in possession changes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        prev_possession_team = self.possession_states.get('last_possession_team', -1)
        current_possession_team = observation['ball_owned_team']
        
        for rew_index in range(len(reward)):
            if current_possession_team != prev_possession_team and current_possession_team != -1:
                components["possession_change_reward"][rew_index] = self.possession_change_reward
                reward[rew_index] += components["possession_change_reward"][rew_index]
        
        # Save the team currently in possession for comparison in the next step.
        self.possession_states['last_possession_team'] = current_possession_team
        
        return reward, components

    def step(self, action):
        """Takes a step in the environment, adding components to info."""
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
