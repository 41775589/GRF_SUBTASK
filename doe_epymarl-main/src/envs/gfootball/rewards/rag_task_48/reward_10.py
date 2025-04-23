import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes from midfield to create scoring opportunities."""

    def __init__(self, env):
        """Initialize the environment and reward mechanism details."""
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward = 0.1
        
    def reset(self):
        """Reset the environment and auxiliary data for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize and save the state of the environment."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize and load the state of the environment."""
        from_pickle = self.env.set_state(state)
        return from_pickle
        
    def reward(self, reward):
        """Calculate the custom reward based on the high passes from midfield."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0, 0.0]}  # Assume two agents are playing

        # Check if observations have valid data
        if observation is None:
            return reward, components

        # Iterate over each agent's observation
        for rew_index, obs in enumerate(observation):
            # Only reward high passes from midfield
            if obs['ball_owned_team'] == 1 and obs['ball'][0] > -0.3 and obs['ball'][0] < 0.3:
                components["pass_reward"][rew_index] = self.pass_quality_reward
                reward[rew_index] += components["pass_reward"][rew_index] * obs['ball'][2] * 5  # modify reward by ball height (z component)

        return reward, components

    def step(self, action):
        """Execute a step in the environment, calculate reward, and return results."""
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
