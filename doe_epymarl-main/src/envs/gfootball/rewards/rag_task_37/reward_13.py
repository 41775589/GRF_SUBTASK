import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering passing under pressure,
    focusing on Short Pass, High Pass, and Long Pass.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_efficiency_counter = np.zeros(10, dtype=int)  # Track efficiency per agent
        self.pressure_threshold = 0.2  # Define pressure as being within this distance from an opponent
        self.pass_reward_coefficient = 0.1  # Reward increment for each successful pass

    def reset(self):
        self.pass_efficiency_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_efficiency_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_efficiency_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] in [0, 1]:  # Check if either team owns the ball
                if obs['active'] == obs['ball_owned_player']:  # Validate active player has the ball
                    # Check proximity of opponents to add pressure-based passing rewards
                    own_player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]
                    opponent_team = 'right_team' if obs['ball_owned_team'] == 0 else 'left_team'
                    
                    # Calculate distance to each opponent player to measure pressure
                    distances = np.linalg.norm(obs[opponent_team] - own_player_pos, axis=1)
                    pressure = np.any(distances < self.pressure_threshold)

                    if pressure:
                        # Reward for successful pass under pressure, modified by pass types
                        active_sticky_actions = obs['sticky_actions']
                        short_pass = active_sticky_actions[9]  # Assuming index 9 is for Short Pass
                        high_pass = active_sticky_actions[8]  # Assuming index 8 is for High Pass
                        long_pass = active_sticky_actions[7]  # Assuming index 7 is for Long Pass
                        
                        # Adding rewards for specific pass types under pressure
                        pass_multiplier = 1.2 if short_pass else 1.5 if long_pass else 1.8 if high_pass else 1
                        components["passing_reward"][i] = self.pass_reward_coefficient * pass_multiplier
                        reward[i] += components["passing_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
