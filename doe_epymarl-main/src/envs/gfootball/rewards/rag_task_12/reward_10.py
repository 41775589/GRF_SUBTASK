import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that emphasizes midfield and advanced defending skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.1
        self._long_pass_reward = 0.1
        self._dribble_reward = 0.05
        self._sprint_reward = 0.05
        self._stop_sprint_reward = 0.025

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward,
            "high_pass_reward": 0.0,
            "long_pass_reward": 0.0,
            "dribble_reward": 0.0,
            "sprint_reward": 0.0,
            "stop_sprint_reward": 0.0
        }

        if observation is None:
            return reward, components
        
        # Reward for successful high passes and long passes
        if observation['game_mode'] == 3:  # Assuming 3 corresponds to a pass mode
            ball_pass_distance = np.linalg.norm(observation['ball_direction'][:2])
            if ball_pass_distance > 0.5:  # assuming this threshold defines a long pass
                components['long_pass_reward'] += self._long_pass_reward
            else:
                components['high_pass_reward'] += self._high_pass_reward

        # Reward for dribbling under pressure
        if observation['sticky_actions'][9] == 1:  # Assuming index 9 is dribbling action
            components['dribble_reward'] += self._dribble_reward

        # Reward for sprinting effectively
        if observation['sticky_actions'][8] == 1:  # Assuming index 8 is sprinting action
            components['sprint_reward'] += self._sprint_reward
        elif observation['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] == 1:
            components['stop_sprint_reward'] += self._stop_sprint_reward

        # Update sticky actions counter
        self.sticky_actions_counter = observation['sticky_actions']
        
        # Compute the final reward by summing all components
        total_reward = sum(components.values())
        return total_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Append individual reward components to info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
