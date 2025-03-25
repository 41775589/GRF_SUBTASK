import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for sprint actions to improve defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_rewards'] = self.sprint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_rewards = from_pickle['sprint_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for idx in range(len(reward)):
            obs = observation[idx]
            play_mode = obs['game_mode']
            controlled_player = obs['active']
            if play_mode == 0 and obs['sticky_actions'][8] == 1:  # Checking if sprint action is active
                # Reward sprint action increases as the game progresses without sprint
                sprint_count = self.sprint_rewards.get(idx, 0) + 1
                self.sprint_rewards[idx] = sprint_count
                sprint_reward = 0.1 / max(1, sprint_count)  # Reduce reward as sprint actions accumulate
                components["sprint_reward"][idx] = sprint_reward
                reward[idx] += sprint_reward

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
