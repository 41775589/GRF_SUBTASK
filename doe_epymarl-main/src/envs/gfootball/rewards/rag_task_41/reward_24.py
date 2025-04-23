import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages offensive play and skill learning, particularly in attacking and creative scenarios 
    under match-like defensive pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds and rewards for relevant match situations
        self.goal_approach_reward = 0.2
        self.goal_threshold = 0.7  # Threshold for rewarding approach towards opponent's goal
        self.passing_reward = 0.1
        self.possession_reward = 0.05
        # Initialize state to track own team's possession for reward computation
        self.possession_state = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_state = {}
        return super().reset()

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

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_approach_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward),
                      "possession_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward for moving the ball close to opposing goal
            if o['ball'][0] > self.goal_threshold and o['ball_owned_team'] == 1:
                components["goal_approach_reward"][rew_index] = self.goal_approach_reward
            
            # Reward for successful passes and possession in the attacking half
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0:
                current_possession = self.possession_state.get(rew_index, 0)
                self.possession_state[rew_index] = current_possession + self.possession_reward
                components["possession_reward"][rew_index] = self.possession_state[rew_index]

            reward[rew_index] += sum(components[c][rew_index] for c in components)

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.possession_state
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.possession_state = from_pickle['CheckpointRewardWrapper']
        return from_pickle
