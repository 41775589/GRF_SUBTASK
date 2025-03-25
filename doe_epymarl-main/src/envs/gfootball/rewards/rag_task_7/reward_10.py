import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles with timing and precision."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_tackle_reward = 1.0  # Reward for successful sliding tackle
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track usage of sticky actions

    def reset(self):
        """Reset the environment state and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include state information about the reward wrapper for environment state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state from a saved state, including specific state from this wrapper."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on sliding tackle actions under high-pressure situations."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward}

        assert len(reward) == len(observation)

        components = {
            "base_score_reward": reward.copy(),
            "sliding_tackle_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_index = o['active']

            # Detect sliding tackle (action index 0 represents sliding tackle in this scenario)
            # Assuming sticky_actions[0] denotes sliding tackle engagement
            if o['sticky_actions'][0] == 1:
                if self._is_high_pressure(o):
                    components["sliding_tackle_reward"][rew_index] += self.sliding_tackle_reward
                    reward[rew_index] += components["sliding_tackle_reward"][rew_index]

            # Update sticky actions usage
            self.sticky_actions_counter += o['sticky_actions']

        return reward, components

    def _is_high_pressure(self, observation):
        """Determine if the current game state means high-pressure for the player."""
        # Example: If the opposing team players are within a certain proximity
        proximity_threshold = 0.2
        player_pos = observation['right_team'][observation['active']]
        opponent_team = 'left_team' if observation['ball_owned_team'] == 1 else 'right_team'
        for opponent in observation[opponent_team]:
            if np.linalg.norm(opponent - player_pos) < proximity_threshold:
                return True
        return False

    def step(self, action):
        """Take action and compute new state and reward."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Updating sticky actions counter into observation info for debugging
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
