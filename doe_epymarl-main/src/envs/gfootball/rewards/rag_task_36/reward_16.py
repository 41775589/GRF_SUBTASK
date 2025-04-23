import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic dribbling and transformation-based reward."""

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
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Initialize components dictionary for reward breakdown
            components.setdefault('dribble_reward', [0.0, 0.0])
            components.setdefault('position_transition_reward', [0.0, 0.0])

            dribbling = o['sticky_actions'][9]  # Index for dribble action

            # Encourage dribbling by giving rewards when dribbling is active
            if dribbling:
                components['dribble_reward'][i] += 0.05
            
            # Encourage dynamic transition by rewarding changes in player's y-pos
            if 'right_team' in o:
                active_player_y_position = o['right_team'][o['active']][1]
                # Reward for significant pos changes since the last check (~0.25 field units)
                if abs(self.sticky_actions_counter[i] - active_player_y_position) > 0.1:
                    components['position_transition_reward'][i] += 0.1
                    self.sticky_actions_counter[i] = active_player_y_position

            # Combine components to form the final reward for each agent
            reward[i] += components['dribble_reward'][i] + components['position_transition_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
