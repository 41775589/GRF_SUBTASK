import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a positional reward based on defensive coordination 
    between agents near the penalty area in high-stress defensive scenarios.
    """
    def __init__(self, env):
        super().__init__(env)
        self.penalty_area_threshold = 0.2  # Threshold to define the penalty area on x-axis
        self.coverage_reward = 1.0          # Reward for covering positions appropriately
        # Tracks positions to avoid multiple rewards in the same area without significant movement
        self.covered_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['covered_zones'] = self.covered_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.covered_zones = from_pickle['covered_zones']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)
        components = {"base_score_reward": reward.copy(),
                      "defensive_coverage_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if 'right_team' in o:
                for i, pos in enumerate(o['right_team']):
                    if abs(pos[0]) > 1 - self.penalty_area_threshold:
                        if rew_index not in self.covered_zones:
                            self.covered_zones[rew_index] = []
                        unique_position = (pos[0], pos[1])
                        # Ensure this position isn't already covered
                        if unique_position not in self.covered_zones[rew_index]:
                            components["defensive_coverage_reward"][rew_index] = self.coverage_reward
                            self.covered_zones[rew_index].append(unique_position)
                            reward[rew_index] += components["defensive_coverage_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Update sticky actions as they are taken
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
