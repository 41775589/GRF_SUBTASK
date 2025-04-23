import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for optimizing shooting angles and timing under pressure near the goal."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_angle_reward = 0.5  # Reward for shooting at optimal angle
        self.timing_reward = 0.3          # Reward for shooting at optimal timing
        self.pressure_reward = 0.2        # Reward for dealing with pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "shooting_angle_reward": [0.0] * len(reward),
                      "timing_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Calculate distance to opponent's goal
            goal_y_position = 0.0
            player_x, player_y = o['right_team'][o['active']]
            distance_to_goal = abs(player_x - 1.0)

            # Reward shooting from optimal shooting angles (closer to the center of the goal)
            optimal_y_shooting_range = -0.2 <= player_y <= 0.2
            if optimal_y_shooting_range and o['ball_owned_player'] == o['active']:
                components["shooting_angle_reward"][rew_index] = self.shooting_angle_reward

            # Reward shooting at optimal timing (less distance to goal better timing assumed)
            if distance_to_goal < 0.3 and o['ball_owned_player'] == o['active']:
                components["timing_reward"][rew_index] = self.timing_reward

            # Reward managing high pressure (more opponents nearby means more pressure)
            opponent_distances = [np.linalg.norm(np.array(o['right_team'][o['active']]) - opp_pos) 
                                  for opp_pos in o['left_team']]
            high_pressure = any(distance < 0.1 for distance in opponent_distances)
            if high_pressure and o['ball_owned_player'] == o['active']:
                components["pressure_reward"][rew_index] = self.pressure_reward

            # Summing up the rewards
            reward[rew_index] += sum(components[key][rew_index] for key in components.keys())

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
