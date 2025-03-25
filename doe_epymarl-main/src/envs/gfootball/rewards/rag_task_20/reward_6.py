import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for offensive strategies and team coordination, focusing on openings,
    defense breaking, passing and positioning, and shooting.
    """

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
        components = {"base_score_reward": reward.copy(),
                      "offensive_strategy_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_score_rew = components["base_score_reward"][rew_index]

            # Additional reward for ball progression towards the opponent's goal.
            if o['ball'][0] > 0:  # Ball is on the opponent's side
                components["offensive_strategy_reward"][rew_index] += 0.1 * (o['ball'][0] + 1)  # Reward ball progression

            # Additional reward for passing or positioning advantage
            if o['ball_owned_team'] == 0 and o['designated'] == o['active']:
                components["offensive_strategy_reward"][rew_index] += 0.05

            # Summing the components for the final reward for this agent
            reward[rew_index] = base_score_rew + components["offensive_strategy_reward"][rew_index]

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
