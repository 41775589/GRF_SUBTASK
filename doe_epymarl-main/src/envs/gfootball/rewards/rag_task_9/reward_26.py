import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on rewarding offensive skills like passing, shooting,
    dribbling, and sprinting to create scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for different skills
        self.passing_reward = 0.1
        self.shooting_reward = 0.2
        self.dribbling_reward = 0.1
        self.sprinting_reward = 0.05
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            active_player_sticky_actions = o['sticky_actions']

            # Reward players for performing specific skills effectively
            if active_player_sticky_actions[1] or active_player_sticky_actions[2]:  # Short or Long Pass
                components["passing_reward"][i] = self.passing_reward
            if active_player_sticky_actions[8]:  # Dribble
                components["dribbling_reward"][i] = self.dribbling_reward
            if active_player_sticky_actions[0] or active_player_sticky_actions[4]:  # Shot
                components["shooting_reward"][i] = self.shooting_reward
            if active_player_sticky_actions[9]:  # Sprint
                components["sprinting_reward"][i] = self.sprinting_reward

            # Total skill reward calculation
            reward[i] += (components["passing_reward"][i] +
                          components["dribbling_reward"][i] +
                          components["shooting_reward"][i] +
                          components["sprinting_reward"][i])

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
            for k, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{k}"] = action
        return observation, reward, done, info
