import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards precise shooting from central positions on the field."""

    def __init__(self, env):
        super().__init__(env)
        self.central_position = np.array([0.0, 0.0])  # central position in normalized coordinates
        self.precision_threshold = 0.05  # position error threshold for precision
        self.power_threshold = 0.2  # threshold for shot power
        self.max_reward = 1.0  # maximum reward for a perfect shot
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        state = self.env.set_state(state)
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward),
            "power_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] != 0:
                # Reward only when in normal play mode
                continue

            ball_position = o['ball'][:2]  # Horizontal ball position
            ball_power = np.linalg.norm(o['ball_direction'][:2])  # Magnitude of ball direction as proxy for power

            # Calculate distance from ball to the central position on the field
            position_error = np.linalg.norm(ball_position - self.central_position)

            # Check if the shot is precise and powerful enough
            if position_error < self.precision_threshold:
                components['precision_reward'][rew_index] = \
                    (self.precision_threshold - position_error) / self.precision_threshold * self.max_reward
            
            if ball_power > self.power_threshold:
                components['power_reward'][rew_index] = \
                    (ball_power - self.power_threshold) / (1 - self.power_threshold) * self.max_reward

            # Aggregate reward values
            reward[rew_index] += components['precision_reward'][rew_index] + components['power_reward'][rew_index]

        return reward, components

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
