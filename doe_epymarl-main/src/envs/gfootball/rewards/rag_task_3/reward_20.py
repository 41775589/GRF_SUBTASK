import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting practice and accuracy reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_positions = {
            -1: {"count": 0, "threshold": 5, "reward": 0.5},
            0: {"count": 0, "threshold": 5, "reward": 0.5},
            1: {"count": 0, "threshold": 5, "reward": 0.5}
        }
        # Dict to hold shooting accuracy from different player positions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_positions = {k: {"count": 0, "threshold": 5, "reward": 0.5} for k in self.shooting_positions}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.shooting_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_positions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_accuracy_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components
          
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if a player scores
            if reward[rew_index] == 1:
                # Reward adjustments for if the player scored
                bal_pos = o['ball_owned_team']
                if bal_pos in self.shooting_positions and self.shooting_positions[bal_pos]['count'] < self.shooting_positions[bal_pos]['threshold']:
                    components["shot_accuracy_reward"][rew_index] = self.shooting_positions[bal_pos]['reward']
                    reward[rew_index] += components["shot_accuracy_reward"][rew_index]
                    self.shooting_positions[bal_pos]['count'] += 1

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
