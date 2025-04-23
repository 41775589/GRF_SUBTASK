import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on ball clearance from defensive zones."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.goal_zone_threshold = -0.8  # Threshold for considering in defensive zone
        self.clearance_reward_multiplier = 5  # Reward for moving the ball out of the defensive zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state of the environment, including wrapper-specific state."""
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        """Sets the state of the environment from saved state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Computes additional reward for clearing the ball from defensive zones."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # Only consider the x position of the ball

            # if the team owning the ball is our team and the ball is in our defensive zone
            if o['ball_owned_team'] == 0 and ball_position < self.goal_zone_threshold:
                # Calculate the distance the ball is moved towards midfield or opponent's side
                movement_towards_midfield = max(0, o['ball'][0] - self.goal_zone_threshold)
                clearance_reward = self.clearance_reward_multiplier * movement_towards_midfield
                components["clearance_reward"][rew_index] += clearance_reward
                reward[rew_index] += clearance_reward

        return reward, components

    def step(self, action):
        """Advances the environment by one timestep."""
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
