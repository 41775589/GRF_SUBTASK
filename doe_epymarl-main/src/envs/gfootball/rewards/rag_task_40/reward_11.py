import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defense and counterattack capabilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of the environment including custom wrapper state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment including custom wrapper state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on defensive plays and initiating counterattacks."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            previous_ball_owned = o.get('ball_owned_team', -1)
            current_ball_owned = o['ball_owned_team']

            # Reward for successful defense: if previously opponent had the ball but now it's either neutral or with us.
            if previous_ball_owned == 1 and current_ball_owned in [0, -1]:
                components['defense_reward'][rew_index] = 0.5
            reward[rew_index] += components['defense_reward'][rew_index]

            # Reward for initiating counterattack: if we just got the ball and are moving forward toward the opponent's goal.
            if current_ball_owned == 0 and o['ball'][0] > o['left_team'][o['active']][0]:
                reward[rew_index] += 0.3

        return reward, components

    def step(self, action):
        """Step the environment and apply modified reward function."""
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
