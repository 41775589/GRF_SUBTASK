import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for mastering high passes and heading for aerial challenges and ball clearance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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

        # Check for high pass effectiveness, rewarded when ball z is above a certain threshold
        if z_ball > 0.2:  # Threshold for determining an aerial pass or clearance
            high_pass_reward = 1.0
            enhanced_reward += high_pass_reward
            components['high_pass_reward'] = [high_pass_reward]
        else:
            components['high_pass_reward'] = [0.0]

        # Additional reward for heading challenges when ball is high
        if observation[0]['ball_owned_player'] == observation[0]['active'] and z_ball > 0.15:
            header_challenge_reward = 2.0
            enhanced_reward += header_challenge_reward
            components['header_challenge_reward'] = [header_challenge_reward]
        else:
            components['header_challenge_reward'] = [0.0]

        return enhanced_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
