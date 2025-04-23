import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for enhancing standing tackles and regaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self.num_successful_tackles = 0
        self.tackle_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the tracking metrics."""
        self.num_successful_tackles = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the current state along with wrapper-specific data."""
        to_pickle['CheckpointRewardWrapper'] = {'num_successful_tackles': self.num_successful_tackles}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state along with wrapper-specific data."""
        from_pickle = self.env.set_state(state)
        self.num_successful_tackles = from_pickle['CheckpointRewardWrapper']['num_successful_tackles']
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on the successful tackles made."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Evaluate if a tackle was successful
        for i, o in enumerate(observation):
            if o['game_mode'] == 3 and o['ball_owned_team'] != 1:  # If free kick against opponent team (success)
                self.num_successful_tackles += 1
                components["tackle_reward"][i] = self.tackle_reward
                reward[i] += components["tackle_reward"][i]

        return reward, components
       
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Logging additional information
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
