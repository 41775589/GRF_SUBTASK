import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes teamwork and defensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_success_threshold = 0.2
        self.defensive_rewards = {}
        self.num_players = 5  # Assuming 5 players as common team configuration

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for index, player_obs in enumerate(observation):
            components["defensive_reward"][index] = self.calculate_defensive_reward(player_obs)
            reward[index] += components["defensive_reward"][index]
        return reward, components

    def calculate_defensive_reward(self, obs):
        """ Calculate defensive rewards based on ball proximity and defensive positioning """
        if obs['ball_owned_team'] == 0 and obs['active'] == obs['ball_owned_player']:
            distance_to_goal = abs(obs['ball'][0] + 1)  # Normalizing position, -1 is goal line for left team defense
            if distance_to_goal < self.defensive_success_threshold:
                return 0.1  # Reward for keeping the ball near the opponent's goal
        elif obs['ball_owned_team'] == 1:  # If opponents have the ball
            controlled_players_positions = obs['left_team'] if obs['ball_owned_team'] == 1 else obs['right_team']
            ball_position = obs['ball'][:2]
            average_distance_to_ball = np.mean([np.linalg.norm(player_pos - ball_position) for player_pos in controlled_players_positions])
            if average_distance_to_ball < self.defensive_success_threshold:
                return 0.05  # Reward for tight defensive formation around the ball
        return 0.0

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
