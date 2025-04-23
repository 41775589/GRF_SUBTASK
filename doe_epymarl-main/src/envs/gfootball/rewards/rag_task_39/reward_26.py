import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive clearance task-specific reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Safe range within defensive zones where clearances are measured
        self.defensive_zones = {
            'left': (-1, -0.2),  # left half closer to the goal
            'right': (1, 0.2)  # right half closer to the goal
        }
        # Reward configuration
        self.clearance_reward = 0.5
        self.penalty_for_conceding = -1

    def reset(self):
        """Reset sticky actions and environment for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State serialization logic."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State deserialization logic."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Custom reward function focused on defensive clearances."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "concede_penalty": [0.0] * len(reward)}
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            is_ball_owned = (o['ball_owned_team'] == 0 or o['ball_owned_team'] == 1)
            ball_x_pos = o['ball'][0]

            # Reward for clearing the ball from defensive zones
            if is_ball_owned and (self.defensive_zones['left'][0] <= ball_x_pos <= self.defensive_zones['left'][1] or
                                  self.defensive_zones['right'][0] >= ball_x_pos >= self.defensive_zones['right'][1]):
                components["clearance_reward"][rew_index] = self.clearance_reward
                reward[rew_index] += components["clearance_reward"][rew_index]
            
            # Penalty if opposing team scores
            if o['score'][0] != o['score'][1]:  # Assume there has been a scoring
                goal_conceded = o['left_team' if o['active'] in o['left_team'] else 'right_team']
                components["concede_penalty"][rew_index] = self.penalty_for_conceding if o['ball_owned_team'] != o['active'] and ball_x_pos in goal_conceded else 0
                reward[rew_index] += components["concede_penalty"][rew_index]

        return reward, components

    def step(self, action):
        """Step through environment and adjust reward based on custom logic."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions in the info
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
