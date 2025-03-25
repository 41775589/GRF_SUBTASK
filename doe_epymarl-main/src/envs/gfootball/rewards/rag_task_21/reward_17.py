import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes defensive plays through rewards for ball interception,
    controlling hazardous areas under pressure, and maintaining defensive positioning.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_reward = 0.2
        self.pressure_zone_control_reward = 0.1
        self.defensive_position_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "pressure_zone_reward": [0.0] * len(reward),
                      "defensive_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            player_pos = o['right_team'][o['active']] if o['active'] is not None and o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            ball_pos = o['ball'][:2]
            
            # Check for interception
            if o['ball_owned_team'] != -1 and np.linalg.norm(ball_pos - player_pos) < 0.03:  # Arbitrary threshold for 'nearness'
                components['interception_reward'][rew_index] = self.intercept_reward
            reward[rew_index] += components['interception_reward'][rew_index]

            # Check for maintaining defensive positioning
            if player_pos[0] < -0.5:  # Player needs to be in defensive half
                components['defensive_position_reward'][rew_index] = self.defensive_position_reward
            reward[rew_index] += components['defensive_position_reward'][rew_index]

            # Handle controlled pressure situations
            if o['ball_owned_team'] == 0 and np.linalg.norm(player_pos - ball_pos) < 0.1:
                components['pressure_zone_reward'][rew_index] = self.pressure_zone_control_reward
            reward[rew_index] += components['pressure_zone_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
