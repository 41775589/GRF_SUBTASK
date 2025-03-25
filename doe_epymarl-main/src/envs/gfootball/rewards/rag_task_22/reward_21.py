import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for sprinting and defensive positioning."""

    def __init__(self, env):
        super().__init__(env)
        # Count of actions focusing on sprinting and defensive coverage
        self.sprint_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sprint_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_actions_counter'] = self.sprint_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_actions_counter = from_pickle['sprint_actions_counter']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "sprint_boost": [0.0, 0.0]}
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            # Encourage sprint when appropriate
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # action_sprint index is 8
                # Increment counter when sprinting
                self.sprint_actions_counter[idx] += 1

                # Calculate sprint boost reward component dynamically, depending on game state
                sprint_boost = 0.02 * (self.sprint_actions_counter[idx] / 10)
                components['sprint_boost'][idx] = sprint_boost
                reward[idx] += sprint_boost

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        # Unpack reward components for logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
