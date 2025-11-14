import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards proper use of sliding tackles, stop moving, and stop sprint commands."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_successful = 0.3  # Reward when a sliding tackle is successfully performed
        self._stop_actions_effective = 0.2  # Effective use of stop moving and stop sprint

    def reset(self):
        """Reset the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state to store for pickling or logging."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state when loading from a pickled state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Reward function that focuses on sliding tackles and effective stopping."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0],
                      "stop_reward": [0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_sliding = o['sticky_actions'][9]  # Sliding action index
            is_stopping = o['sticky_actions'][6]  # Stop moving index
            is_not_sprinting = not o['sticky_actions'][8]  # Stop sprint index

            # Reward for sliding action if it’s effectively dispossessing the opponent
            if is_sliding and o['ball_owned_team'] == 1:
                components['tackle_reward'][rew_index] = self._tackle_successful
                reward[rew_index] += components['tackle_reward'][rew_index]

            # Reward effective use of stop moving and sprint together with tactical positioning
            if is_stopping and is_not_sprinting:
                components['stop_reward'][rew_index] = self._stop_actions_effective
                reward[rew_index] += components['stop_reward'][rew_index]
                
        return reward, components

    def step(self, action):
        """Override the base step function to include adjusted rewards based on the custom logic."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
