import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning and transition reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter and underlying environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include the wrapper's state in the pickled information."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from the pickled information."""
        from_pickle = self.env.set_state(state)
        # Nothing specific to restore since no internal state is modified
        return from_pickle

    def reward(self, reward):
        """Custom reward function emphasizing strategic positioning and effective transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Enhance defensive positioning
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                defense_position_index = np.argmin(o['left_team'][:, 0])  # Closest to own goal
                if o['active'] == defense_position_index:
                    reward[rew_index] += 0.2

            # Reward transition to counterattack
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if 'left_team_direction' in o:
                    player_velocity = np.linalg.norm(o['left_team_direction'][o['active']])
                    # Reward increases as the controlled player's speed increases
                    reward[rew_index] += 0.5 * player_velocity

        return reward, components

    def step(self, action):
        """Process a step in the environment, augmenting reward information in debug."""
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
