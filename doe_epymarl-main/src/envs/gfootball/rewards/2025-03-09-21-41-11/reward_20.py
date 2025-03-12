import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive strategies for football games."""
    def __init__(self, env):
        super().__init__(env)
        self.shooting_range = 0.2
        self.pass_bonus = 0.1
        self.dribble_bonus = 0.05
        self.successful_shots = 0
        self.passes_completion = 0
        self.dribbles_made = 0

    def reset(self):
        self.successful_shots = 0
        self.passes_completion = 0
        self.dribbles_made = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_reward = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": 0.0,
            "passing_reward": 0.0,
            "dribbling_reward": 0.0
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            ball_pos = o['ball']
            if np.linalg.norm(ball_pos[:2]) > (1 - self.shooting_range) and o['ball_owned_player'] == o['active']:
                self.successful_shots += 1
                components['shooting_reward'] += 1.0

            if o['sticky_actions'][9]:  # 'action_dribble' is active
                self.dribbles_made += 1
                new_reward[i] += self.dribble_bonus
                components['dribbling_reward'] += self.dribble_bonus

            if o['sticky_actions'][0] or o['sticky_actions'][4]:  # 'action_left' or 'action_right'
                self.passes_completion += 1
                new_reward[i] += self.pass_bonus
                components['passing_reward'] += self.pass_bonus

        return new_reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info['final_reward'] = sum(reward)

        # Traverse the components dictionary and add each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
