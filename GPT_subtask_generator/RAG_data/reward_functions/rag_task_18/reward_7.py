import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that includes a reward for the central midfield's performance in managing transitions and pace."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_ball_control_rewards = np.zeros(2, dtype=float)
        self.tactical_positioning_rewards = np.zeros(2, dtype=float)
        self.pace_adjustment_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_ball_control_rewards.fill(0)
        self.tactical_positioning_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_midfield_control'] = self.midfield_ball_control_rewards.tolist()
        to_pickle['CheckpointRewardWrapper_tactical_position'] = self.tactical_positioning_rewards.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_ball_control_rewards = np.array(from_pickle['CheckpointRewardWrapper_midfield_control'])
        self.tactical_positioning_rewards = np.array(from_pickle['CheckpointRewardWrapper_tactical_position'])
        return from_pickle

    def reward(self, reward):
        """Customize rewards based on midfield control and tactical positioning."""
        observation = self.env.unwrapped.observation()

        new_rewards = reward.copy()
        reward_components = {"base_score_reward": reward.copy(), 
                             "midfield_control_reward": [0.0, 0.0],
                             "tactical_positioning_reward": [0.0, 0.0]}

        for player_idx, (rew, obs) in enumerate(zip(reward, observation)):
            # Control-based reward: If our midfield player has the ball
            if obs['ball_owned_team'] == 0 and obs['active'] in (4, 5):  # central midfield roles
                self.midfield_ball_control_rewards[player_idx] += 0.05
                new_rewards[player_idx] += self.midfield_ball_control_rewards[player_idx]

            # Positioning-based reward: Encouraging midfielders to stay in the central part of the field
            if obs['active'] in (4, 5):  # central midfield roles
                midfield_y_position = obs['left_team'][obs['active']][1]
                # Encourage being near the y=0 position
                positioning_reward = self.pace_adjustment_factor * (1 - abs(midfield_y_position))
                self.tactical_positioning_rewards[player_idx] += positioning_reward
                new_rewards[player_idx] += self.tactical_positioning_rewards[player_idx]

            reward_components["midfield_control_reward"][player_idx] = self.midfield_ball_control_rewards[player_idx]
            reward_components["tactical_positioning_reward"][player_idx] = self.tactical_positioning_rewards[player_idx]

        return new_rewards, reward_components

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
