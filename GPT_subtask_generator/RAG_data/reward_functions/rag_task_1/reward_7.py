import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic game phase-oriented reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize the reward modification factors
        self.quick_attack_reward = 0.3
        self.possession_change_reward = 0.2
        self.dynamic_adaptation_reward = 0.1

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
        # Extract the current observation from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "quick_attack_reward": [0.0] * len(reward),
                      "possession_change_reward": [0.0] * len(reward),
                      "dynamic_adaptation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Assume the observation is structured correctly
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            previous_ball_pos = self.previous_observation[rew_index]['ball'] if self.previous_observation else None

            # Quick attack: evaluate the possession and position change
            if o['ball_owned_team'] == 1 and (previous_ball_pos is None or np.linalg.norm(o['ball'] - previous_ball_pos) > 0.05):
                components["quick_attack_reward"][rew_index] = self.quick_attack_reward
                reward[rew_index] += self.quick_attack_reward

            # Possession change
            if self.previous_observation and o['ball_owned_team'] != self.previous_observation[rew_index]['ball_owned_team']:
                components["possession_change_reward"][rew_index] = self.possession_change_reward
                reward[rew_index] += self.possession_change_reward

            # Dynamic adaptation to different game modes
            if o['game_mode'] != self.previous_observation[rew_index]['game_mode']:
                components["dynamic_adaptation_reward"][rew_index] = self.dynamic_adaptation_reward
                reward[rew_index] += self.dynamic_adaptation_reward

        # Store the current observation for the next reward computation
        self.previous_observation = observation.copy()

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions used
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
