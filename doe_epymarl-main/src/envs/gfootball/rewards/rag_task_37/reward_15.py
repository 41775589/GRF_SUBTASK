import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for advanced ball control and passing under pressure scenarios."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.control_under_pressure_reward = 0.1

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
                      "passing_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            own_ball_team = o['ball_owned_team'] == 0
            nearby_opponents = self.detect_opponents(o['left_team'], o['ball'])

            # Check if the player performs a pass when surrounded by opponents
            is_pass = o['sticky_actions'][5] or o['sticky_actions'][6]
            if own_ball_team and nearby_opponents and is_pass:
                components["passing_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["passing_reward"][rew_index]

            # Control under pressure: When possessing the ball near opponents without losing it
            if own_ball_team and nearby_opponents and not is_pass:
                components["control_under_pressure_reward"][rew_index] = self.control_under_pressure_reward
                reward[rew_index] += components["control_under_pressure_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def detect_opponents(self, team_positions, ball_position):
        """Check if there are opponents near the ball."""
        threshold = 0.1  # Approximate distance indicating 'closeness'
        return any(np.linalg.norm(player_pos - ball_position[:2]) < threshold for player_pos in team_positions)
