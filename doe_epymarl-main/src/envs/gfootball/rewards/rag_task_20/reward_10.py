import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies focusing on team coordination and position gameplay."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Calculate rewards based on strategy
        for rew_index, obs in enumerate(observation):
            # Add incentives for maintaining ball possession and advancing towards the goal
            if obs['ball_owned_team'] == 1 and obs['ball'][0] > 0:  # Team 1 has the ball and is moving right
                components["positioning_reward"][rew_index] = 0.1 * obs['ball'][0]  # Reward for positioning towards opponent's goal
                reward[rew_index] += components["positioning_reward"][rew_index]
            
            # Reward for successful passes
            if obs['game_mode'] in {1, 3, 4, 6}:  # Game modes corresponding to kick-offs, free kicks, corners, penalties
                components["passing_reward"][rew_index] = 0.05
                reward[rew_index] += components["passing_reward"][rew_index]

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
