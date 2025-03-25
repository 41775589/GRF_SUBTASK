import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive synergizing reward for coordination near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.teammate_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.teammate_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = self.env.get_state(to_pickle)
        state_data['CheckpointRewardWrapper'] = self.teammate_positions
        return state_data

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.teammate_positions = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_synergy_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_synergy_reward"][rew_index] = self.calculate_defensive_synergy(o, rew_index)
            reward[rew_index] += components["defensive_synergy_reward"][rew_index]

        return reward, components

    def calculate_defensive_synergy(self, observation, agent_index):
        # Extract key features for the defensive roles
        penalty_area_threshold = 0.3
        defensive_synergy_bonus = 0.0
        teammate_close_bonus = 0.1
        ball_position = observation['ball'][:2]  # We consider the x, y coordinates

        # If the ball is in the penalty area.
        if abs(ball_position[0]) > (1 - penalty_area_threshold):
            if 'left_team' in observation:
                team_positions = observation['left_team'] if ball_position[0] < 0 else observation['right_team']
            elif 'right_team' in observation:
                team_positions = observation['right_team'] if ball_position[0] > 0 else observation['left_team']

            self.teammate_positions[agent_index] = team_positions[observation['active']]
            for teammate in team_positions:
                if np.linalg.norm(teammate - self.teammate_positions[agent_index]) < penalty_area_threshold:
                    defensive_synergy_bonus += teammate_close_bonus

        return defensive_synergy_bonus
    
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
