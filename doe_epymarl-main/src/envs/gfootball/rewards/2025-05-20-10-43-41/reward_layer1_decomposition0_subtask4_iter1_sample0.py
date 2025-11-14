import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for mastering high passes and heading for aerial challenges and ball clearance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Introducing new reward components
        self.high_pass_bonus = 0.5
        self.header_bonus = 0.7

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        enhanced_reward = reward.copy()
        components = {'base_score_reward': reward.copy()}

        z_ball = observation[0]['ball'][2]  # Get z-coordinate of the ball

        # Reward for effectively handling high passes 
        if z_ball > 0.3:  # Higher threshold for a more targeted high pass
            enhanced_reward[0] += self.high_pass_bonus
            components['high_pass_bonus'] = [self.high_pass_bonus]
        else:
            components['high_pass_bonus'] = [0.0]

        # Reward for successful heading
        if observation[0]['ball_owned_player'] == observation[0]['active'] and z_ball > 0.25:
            enhanced_reward[0] += self.header_bonus
            components['header_bonus'] = [self.header_bonus]
        else:
            components['header_bonus'] = [0.0]

        return enhanced_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
