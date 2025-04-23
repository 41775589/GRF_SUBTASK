import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense strategic positioning reward based on maintaining a balance between defense and attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Initialize tactical positioning parameters.
        self.defensive_threshold = -0.5  # Threshold to define defensive areas.
        self.attack_threshold = 0.5       # Threshold to define attacking areas.
        self.pivoting_reward = 0.05       # Reward for effective pivoting.
        self.dist_from_def_to_attack = 0.2 # Minimal distance to switch roles effectively.
        self.last_position = None         # Store the last position to compare movement.

    def reset(self):
        """Reset the internal state at the beginning of an episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encapsulate the current state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {
            'last_position': self.last_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.last_position = from_pickle['CheckpointRewardWrapper'].get('last_position')
        return from_pickle

    def reward(self, reward):
        """Compute dense reward for maintaining strategic game balance and good pivoting."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "strategic_positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_position = o['active']

            # Update defensive to attacking transitions
            if self.last_position is not None:
                if self.last_position < self.defensive_threshold <= current_position:
                    if current_position - self.last_position > self.dist_from_def_to_attack:
                        components["strategic_positioning_reward"][rew_index] = self.pivoting_reward

            # Update attacking to defensive transitions
            if self.last_position is not None:
                if self.last_position > self.attack_threshold >= current_position:
                    if self.last_position - current_position > self.dist_from_def_to_attack:
                        components["strategic_positioning_reward"][rew_index] = self.pivoting_reward

            # Update reward based on strategy component
            reward[rew_index] += components["strategic_positioning_reward"][rew_index]

            # Remember the current position for the next reward calculation
            self.last_position = current_position

        return reward, components

    def step(self, action):
        """Apply an action, step the environment, and apply the reward modification."""
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
