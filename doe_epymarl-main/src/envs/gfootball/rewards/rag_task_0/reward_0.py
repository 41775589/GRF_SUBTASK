import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward for offensive strategies focused on shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracks sticky actions for each agent

    def reset(self):
        """Reset the reward wrapper state along with the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper with additional wrapper states."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper from the pickle object."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Custom reward function to encourage effective offensive plays."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_shooting = o['sticky_actions'][9]  # Assuming index 9 is related to shooting action
            effective_dribble = o['sticky_actions'][8]  # Assuming index 8 is dribbling
            is_passing = o['sticky_actions'][7]  # Assuming index 7 is passing

            if is_shooting:
                components["shooting_reward"][rew_index] = 0.1
                reward[rew_index] += components["shooting_reward"][rew_index]
            
            if effective_dribble:
                components["dribbling_reward"][rew_index] = 0.05
                reward[rew_index] += components["dribbling_reward"][rew_index]
            
            if is_passing:
                components["passing_reward"][rew_index] = 0.03
                reward[rew_index] += components["passing_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Step the environment and apply the reward wrapper."""
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
        return observation, reward, done, info
