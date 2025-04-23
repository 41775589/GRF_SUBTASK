import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on dribbling and positional transition skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions engaged by agents
        self.position_change_reward = 0.05  # Reward for changing positions effectively

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add the current state to pickle."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Fold dribbling and positional changes into the reward system."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_transition_reward": [0.0] * len(reward),
            "position_change_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            current_sticky = obs['sticky_actions']

            if 'ball_owned_player' in obs and obs['ball_owned_player'] == obs['active']:
                # Reward for dribbling effectively
                if current_sticky[9]:  # Dribble action engaged
                    components["dribble_transition_reward"][rew_index] += 0.1
                    reward[rew_index] += components["dribble_transition_reward"][rew_index]
                
                # Reward for strategic positioning: dynamic transitioning between offense and defense
                # Calculate reward based on player's proximity to optimal positions depending on ball ownership
                optimal_y_pos = -obs['ball'][1] if obs['ball_owned_team'] == 1 else obs['ball'][1]
                position_factor = abs(obs['left_team'][obs['active']][1] - optimal_y_pos)
                components["position_change_reward"][rew_index] = self.position_change_reward - position_factor
                reward[rew_index] += components["position_change_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take a step using action, and attach additional reward components and update info to the environment's response."""
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
