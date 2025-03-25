import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive synergy reward in high-pressure situations near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward parameters
        self.defensive_zone_threshold = 0.3  # Threshold for extra rewards when agents are close to their own goal
        self.cooperative_defense_reward = 0.2  # Extra reward for cooperative defense

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
                      "defensive_synergy_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            proximity_to_goal_penalty = (abs(o['left_team'][o['active']][0] + 1) < self.defensive_zone_threshold)
            has_ball = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])
            actively_defending = o['sticky_actions'][8] or o['sticky_actions'][9]  # Check if sprinting or dribbling

            if proximity_to_goal_penalty and has_ball and actively_defending:
                components["defensive_synergy_reward"][rew_index] = self.cooperative_defense_reward
                reward[rew_index] += components["defensive_synergy_reward"][rew_index]

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
