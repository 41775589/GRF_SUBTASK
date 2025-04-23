import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to encourage midfield mastery, focusing on ball control in midfield and supporting both defense and attack transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Track midfield control and distribution
        self.midfield_control_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Midfield checkpoints can be dynamically adjusted based on observations
        self.midfield_zones = {
            'central': [[-0.3, 0.3], [-0.15, 0.15]],  # central midfield region
            'wide': [[-0.3, 0.3], [-0.42, -0.20], [0.20, 0.42]]  # wide areas of the midfield
        }
        self.reward_weights = {
            'maintain_midfield_possession': 0.05,
            'transition_support': 0.05
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        to_pickle['CheckpointRewardWrapper'] = {
            'midfield_control_counter': self.midfield_control_counter
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_counter = from_pickle['CheckpointRewardWrapper']['midfield_control_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'midfield_possession': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            ball_pos = obs['ball'][0:2]
            if self.in_midfield(ball_pos, 'central'):
                # Reward based on controlled ball possession in central midfield
                if obs['ball_owned_team'] == 0:  # assuming '0' is the team of the agent
                    components['midfield_possession'][i] = self.reward_weights['maintain_midfield_possession']
                    reward[i] += components['midfield_possession'][i]
                    self.midfield_control_counter += 1

            # Supporting transitions: Check if agents contribute to transitions from defense to attack
            if self.in_midfield(ball_pos, 'wide'):
                if obs['ball_owned_team'] == 0:
                    components['midfield_possession'][i] += self.reward_weights['transition_support']
                    reward[i] += self.reward_weights['transition_support']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def in_midfield(self, position, zone_key):
        """Check if the position is within the specified midfield zone."""
        x_range = self.midfield_zones[zone_key][0]
        if zone_key == 'central':
            y_range = self.midfield_zones[zone_key][1]
            return x_range[0] <= position[0] <= x_range[1] and y_range[0] <= position[1] <= y_range[1]
        else:  # wide
            left_y_range = self.midfield_zones[zone_key][1]
            right_y_range = self.midfield_zones[zone_key][2]
            in_left = x_range[0] <= position[0] <= x_range[1] and left_y_range[0] <= position[1] <= left_y_range[1]
            in_right = x_range[0] <= position[0] <= x_range[1] and right_y_range[0] <= position[1] <= right_y_range[1]
            return in_left or in_right
