import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on offensive strategies including accurate shooting,
    effective dribbling, and different types of passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_coefficient = 0.5
        self.dribbling_coefficient = 0.3
        self.pass_success_coefficient = 0.2

    def reset(self):
        """Resets the environment and rewards."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Returns the game state for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the game state from deserialization."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Enhances base reward by considering offensive plays."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward), "pass_success_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            if o['game_mode'] == 6:  # Check if it's a shooting opportunity (Penalty)
                components["shooting_reward"][idx] = self.shooting_coefficient
                reward[idx] += components["shooting_reward"][idx]
            
            if o['sticky_actions'][9]:  # Dribbling action is active
                components["dribbling_reward"][idx] = self.dribbling_coefficient
                reward[idx] += components["dribbling_reward"][idx]
            
            # Checking for successful passes
            if o['game_mode'] in [2, 5]:  # Game modes for free kick or throw-in
                components["pass_success_reward"][idx] = self.pass_success_coefficient
                reward[idx] += components["pass_success_reward"][idx]

        return reward, components

    def step(self, action):
        """Applies the action, calculates the reward, and returns the next state and info."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        # Add sticky actions to info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = action_state
        return obs, reward, done, info
