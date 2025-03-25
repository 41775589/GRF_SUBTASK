import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper to encourage sprinting and quickly covering distances for better defensive positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_distances = [0.0, 0.0]  # Initialize covered distances for both agents

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_distances = [0.0, 0.0]
        return self.env.reset()

    def get_state(self, to_pickle):
        state = {'covered_distances': self.covered_distances}
        to_pickle.update(state)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.covered_distances = from_pickle['covered_distances']
        return from_pickle

    def reward(self, reward):
        """Reward agents based on the distance covered in sprint mode."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        # Initialize components dictionary to track individual components of reward
        components = {"base_score_reward": reward.copy(), "sprint_distance_reward": [0.0, 0.0]}
        
        for idx, obs in enumerate(observation):
            # Reward sprinting activity to promote faster coverage of the field
            if obs['sticky_actions'][8] == 1:  # Check if the sprint action is active
                player_pos = obs['right_team'][obs['active']] if obs['ball_owned_team'] == 1 else obs['left_team'][obs['active']]
                prev_pos = player_pos if self.covered_distances[idx] == 0 else self.covered_distances[idx]
                dist = np.linalg.norm(np.array(player_pos) - np.array(prev_pos))
                self.covered_distances[idx] = dist
                sprint_reward = dist * 0.1  # Define a coefficient for distance-based reward
                components["sprint_distance_reward"][idx] = sprint_reward
                reward[idx] += sprint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
