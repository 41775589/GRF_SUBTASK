import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for offensive maneuvers and adaptation in different game phases."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_rewards = {}
        self.phase_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_rewards = {}
        self.phase_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['offensive_rewards'] = self.offensive_rewards
        to_pickle['phase_rewards'] = self.phase_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.offensive_rewards = from_pickle['offensive_rewards']
        self.phase_rewards = from_pickle['phase_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_reward": [0.0] * len(reward),
                      "adaptation_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            # Calculate offensive rewards based on ball control and position
            if obs['ball_owned_team'] == 1 and obs['active'] == obs['ball_owned_player']:
                distance_to_goal = 1 - obs['ball'][0]  # assuming goal is at x=1 for the opponent
                components["offensive_reward"][i] = 0.1 * distance_to_goal

            # Calculate adaptation reward based on game mode transitions
            if self.phase_rewards.get(i) is None:
                self.phase_rewards[i] = obs['game_mode']
            elif self.phase_rewards[i] != obs['game_mode']:
                components["adaptation_reward"][i] = 0.2  # reward for adapting to new game phase
                self.phase_rewards[i] = obs['game_mode']

            # Update rewards
            reward[i] += components["offensive_reward"][i] + components["adaptation_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        if obs is not None:
            for agent_obs in obs:
                for i, act in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += act
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
