import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining strategic positioning with effective use of actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        self.initialized_positions = False
        self.winning_margin = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        self.initialized_positions = False
        self.winning_margin = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positional_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Initialize strategic positions once per episode
            if not self.initialized_positions:
                # Halfway lines of the pitch as strategic threshold positions
                self.positional_rewards[rew_index] = 0
                self.winning_margin[rew_index] = 0
                
                # Corner strategic positions
                self.positional_rewards[rew_index] += 0.1 if abs(o['left_team'][rew_index][0]) > 0.8 and abs(o['left_team'][rew_index][1]) > 0.3 else 0

            # Ensuring that the agent actively uses all possible directions
            active_moves = sum(o['sticky_actions'][-8:])
            self.sticky_actions_counter += o['sticky_actions']

            # Reward for using non-redundant, effective directional moves
            components['positional_reward'][rew_index] += 2 * active_moves - self.sticky_actions_counter.sum() * 0.1

            # Reward for maintaining close distance to ball in defense or while attacking
            ball_distance = np.linalg.norm(o['ball'][:2] - o['left_team'][rew_index][:2])
            if ball_distance < 0.1:
                components['positional_reward'][rew_index] += 0.5
            
            # Combine base reward and auxiliary rewards
            reward[rew_index] += components['base_score_reward'][rew_index] + components['positional_reward'][rew_index]

        self.initialized_positions = True

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'positions': self.positional_rewards, 'winning_margin': self.winning_margin}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        loaded_data = from_pickle['CheckpointRewardWrapper']
        self.positional_rewards = loaded_data['positions']
        self.winning_margin = loaded_data['winning_margin']
        return from_pickle

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
