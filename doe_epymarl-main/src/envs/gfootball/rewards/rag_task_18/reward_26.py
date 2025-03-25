import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that aims to enhance midfield synchronization and controlled play pacing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define a simple reward tweak to incentivize midfield control and transition handling
        self.midfield_zone_threshold = 0.2  # arbitrary threshold for midfield zone
        self.midfield_control_reward = 0.05  # reward for controlling the ball in the midfield
        self.pace_control_reward = 0.05  # reward for managing pace (defined by movement speed)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        # No internal state to save or restore in this simple example
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No internal state to save or restore in this simple example
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "midfield_control_reward": [0.0] * len(reward), "pace_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for controlling midfield interaction
            if np.abs(o['ball'][0]) < self.midfield_zone_threshold and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                reward[rew_index] += self.midfield_control_reward
                components["midfield_control_reward"][rew_index] = self.midfield_control_reward

            # Reward for controlled pace management
            speed = np.linalg.norm(o['left_team_direction'][o['active']])
            if speed >= 0.01 and speed <= 0.03:
                reward[rew_index] += self.pace_control_reward
                components["pace_control_reward"][rew_index] = self.pace_control_reward

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
