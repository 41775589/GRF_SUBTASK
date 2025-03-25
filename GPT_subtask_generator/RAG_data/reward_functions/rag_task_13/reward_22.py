import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to enhance defensive actions, focusing on man-marking,
    blocking shots, and stalling forwards through rewards.
    """
    def __init__(self, env):
        super().__init__(env)
        self._num_blocks = 5  # Number of defensive blocks to track for reward
        self._block_reward = 0.2  # Reward for successful defense
        self.defensive_actions = np.zeros(5, dtype=int)  # Counter for defensive actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky action tracker
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_actions': self.defensive_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions = from_pickle['CheckpointRewardWrapper']['defensive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Rewarding based on defensive behavior
            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Defensive game modes (e.g., Free Kick, Corner, etc.)
                if (o['ball_owned_team'] == 0) and (o['active'] in o['left_team_roles']):
                    # Tracking defensive blocks
                    if self.defensive_actions[rew_index] < self._num_blocks:
                        components['defensive_reward'][rew_index] = self._block_reward
                        reward[rew_index] += components['defensive_reward'][rew_index]
                        self.defensive_actions[rew_index] += 1

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
