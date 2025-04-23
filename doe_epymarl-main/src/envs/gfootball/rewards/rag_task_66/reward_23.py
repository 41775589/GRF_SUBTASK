import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for short passes under defensive pressure focusing on ball retention."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_reward = 0.1  # Reward increment for each successful pass under pressure
        self.ball_retention_reward = 0.1  # Reward for retaining ball under pressure
        self.defensive_stability_reward = 0.2  # Additional reward for effective defense positioning
        self.num_players = None
        self.pressure_threshold = 0.3  # Represents the average distance threshold for considering a player under pressure

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
                      "short_pass_reward": [0.0] * len(reward),
                      "ball_retention_reward": [0.0] * len(reward),
                      "defensive_stability_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assuming all observations are structured uniformly 
            if self.num_players is None:
                self.num_players = len(o['left_team'])

            # Calculate pressure based on opponent's proximity
            if o['ball_owned_team'] == 0:  # If left team has the ball
                opponents = o['right_team']
                teammates = o['left_team']
            else:
                opponents = o['left_team']
                teammates = o['right_team']

            ball_pos = o['ball'][:2]
            player_pos = teammates[o['active']]

            distances = np.linalg.norm(opponents - player_pos, axis=1)
            pressure = np.mean(distances < self.pressure_threshold)

            # Reward player for successful short pass under pressure
            if 'action' in o and o['action'] == 'short_pass' and pressure > 0.5:
                components['short_pass_reward'][rew_index] = self.pass_accuracy_reward
                reward[rew_index] += components['short_pass_reward'][rew_index]

            # Reward maintaining possession under pressure
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components['ball_retention_reward'][rew_index] = self.ball_retention_reward
                reward[rew_index] += components['ball_retention_reward'][rew_index]

            # Calculate defensive stability (reward for effective defensive positioning)
            distances_to_ball = np.linalg.norm(teammates - ball_pos, axis=1)
            if np.any(distances_to_ball < self.pressure_threshold):
                components['defensive_stability_reward'][rew_index] = self.defensive_stability_reward
                reward[rew_index] += components['defensive_stability_reward'][rew_index]

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
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
        return observation, reward, done, info
