import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting-specific reward based on positioning and shooting power/accuracy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._reward_for_shooting_position = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encodes the wrapper state into a pickleable format."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Decodes the state from a pickled format and sets the wrapper state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augments the reward based on shooting from central field positions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_position_reward": [0.0] * len(reward)
        }
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate the distance to the center line and reward shooting from near the center
            if o['active'] is not None and 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
                distance_from_center = abs(player_pos[1])  # y-coordinate distance from center line
                components['shooting_position_reward'][rew_index] = max(0, self._reward_for_shooting_position * (0.42 - distance_from_center) / 0.42)

            # Updating the reward list with additional components
            reward[rew_index] += components['shooting_position_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Steps the environment, applies reward wrappers, and returns new observations and rewards."""
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
