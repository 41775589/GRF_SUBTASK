import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds a reward for successful long-range shots from outside the penalty box. """

    def __init__(self, env):
        super().__init__(env)
        self.long_shot_threshold = 0.7  # Defines the distance threshold for a long-range attempt
        self.goal_reward = 1.0  # Reward for scoring a goal
        self.long_shot_reward = 0.5  # Additional reward for scoring from long range
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        modified_rewards = reward.copy()
        components = {'base_score_reward': reward.copy(), 'long_shot_bonus': [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            score_change = o['score'][1] - o['score'][0]  # Assuming player is on team 1
            has_shot_scored = score_change > 0
            ball_x_pos = o['ball'][0]

            # Check if goal was scored and if it was from long range
            if has_shot_scored and ball_x_pos < -self.long_shot_threshold:
                modified_rewards[rew_index] += self.long_shot_reward
                components['long_shot_bonus'][rew_index] = self.long_shot_reward

        return modified_rewards, components

    def step(self, action):
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
