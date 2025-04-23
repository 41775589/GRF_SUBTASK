import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on maintaining ball control under pressure,
    exploiting open spaces, and effective passing.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_sensitive_reward = 0.05
        self.open_space_reward = 0.03
        self.pass_quality_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # There is no internal state maintained by this wrapper so just return
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pressure_sensitive_reward": [0.0] * len(reward),
                      "open_space_reward": [0.0] * len(reward),
                      "pass_quality_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['active']

            # Reward for maintaining ball under pressure
            if o['ball_owned_team'] == 0:  # Assuming 0 is the team of the agent
                enemy_distance = np.min(np.linalg.norm(o['left_team'] - o['ball'], axis=1))
                if enemy_distance < 0.1:  # Arbitrary threshold for "pressure"
                    components["pressure_sensitive_reward"][rew_index] = self.pressure_sensitive_reward
                    reward[rew_index] += components["pressure_sensitive_reward"][rew_index]

            # Reward for exploiting open space
            teammate_distances = np.linalg.norm(o['right_team'] - o['ball'], axis=1)
            if np.all(teammate_distances > 0.2):  # If all teammates are far enough
                components["open_space_reward"][rew_index] = self.open_space_reward
                reward[rew_index] += components["open_space_reward"][rew_index]

            # Reward for effective passing
            # Check for a change in ball ownership to the teammate after an action
            if o['ball_owned_player'] is not active_player_pos and o['ball_owned_team'] == 0:
                components["pass_quality_reward"][rew_index] = self.pass_quality_reward
                reward[rew_index] += components["pass_quality_reward"][rew_index]

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
