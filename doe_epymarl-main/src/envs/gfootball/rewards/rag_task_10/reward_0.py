import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on reinforcing defensive actions such as sliding, stopping dribbles, and blocking the opponent's movements."""

    def __init__(self, env):
        super().__init__(env)
        # Count of defensive moves: sliding, stop-dribble, stop-moving
        self.defensive_actions_count = np.zeros(3, dtype=int)  # [sliding, stop_dribble, stop_moving]
        self.defensive_rewards = [0.2, 0.1, 0.05]  # Custom rewards for each defensive action
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Monitoring sticky actions

    def reset(self):
        """Reset the environment and defensive actions count."""
        self.defensive_actions_count = np.zeros(3, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include the defensive action counts when getting state for pickling."""
        to_pickle['defensive_actions_count'] = self.defensive_actions_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from pickle, updating the defensive action counts."""
        from_pickle = self.env.set_state(state)
        self.defensive_actions_count = from_pickle['defensive_actions_count']
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_reward": [0.0]}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(reward):
            # Defensive actions
            sliding = self.sticky_actions_counter[4]
            stop_dribble = self.sticky_actions_counter[7]
            stop_moving = self.sticky_actions_counter[6]

            # Add rewards for defensive actions
            if sliding > self.defensive_actions_count[0]:
                reward[rew_index] += self.defensive_rewards[0]
                self.defensive_actions_count[0] += 1
            if stop_dribble > self.defensive_actions_count[1]:
                reward[rew_index] += self.defensive_rewards[1]
                self.defensive_actions_count[1] += 1
            if stop_moving > self.defensive_actions_count[2]:
                reward[rew_index] += self.defensive_rewards[2]
                self.defensive_actions_count[2] += 1

        return reward, components

    def step(self, action):
        """Perform an environment step and apply the reward modification."""
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
