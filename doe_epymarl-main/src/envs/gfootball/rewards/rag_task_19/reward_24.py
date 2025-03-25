import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic, focused reward for effective defense and
       midfield control."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoint_reward = 0.1
        self.defensive_checkpoint_reward = 0.2
        self.midfield_zone_threshold = 0.5  # midfield threshold on x-axis
        self.defense_zone_threshold = -0.5  # defense threshold on x-axis for the left team

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": np.array(reward),
                      "midfield_control_reward": np.zeros_like(reward),
                      "defensive_reward": np.zeros_like(reward)}
        
        for idx, o in enumerate(observation):
            ball_pos_x = o['ball'][0]
            ball_owned_team = o['ball_owned_team']
            active_player_x = o['left_team'][ball_owned_team] if ball_owned_team == 0 else o['right_team'][ball_owned_team]

            # Midfield control
            if abs(ball_pos_x) < self.midfield_zone_threshold:
                components["midfield_control_reward"][idx] = self.midfield_checkpoint_reward

            # Effective defense
            if ball_owned_team == 0 and active_player_x < self.defense_zone_threshold:
                components["defensive_reward"][idx] = self.defensive_checkpoint_reward
        
        for key, value_array in components.items():
            reward += value_array

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        np.fill(self.sticky_actions_counter, 0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
