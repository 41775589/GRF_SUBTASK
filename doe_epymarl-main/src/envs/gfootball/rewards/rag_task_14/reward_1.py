import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful defensive actions typical for a sweeper role."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.close_ball_interceptions = 0
        self.deep_defensive_actions = 0
        # reward increments
        self.interception_reward = 0.3
        self.defensive_position_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.close_ball_interceptions = 0
        self.deep_defensive_actions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'close_ball_interceptions': self.close_ball_interceptions,
            'deep_defensive_actions': self.deep_defensive_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        custom_state = from_pickle['CheckpointRewardWrapper']
        self.close_ball_interceptions = custom_state['close_ball_interceptions']
        self.deep_defensive_actions = custom_state['deep_defensive_actions']
        return from_pickle

    def reward(self, reward):
        base_reward = reward.copy()
        observation = self.unwrapped.observation()
        
        if observation is None:
            return base_reward

        obs_left_team = observation['left_team']
        ball_position = observation['ball'][:2]
        for player_pos in obs_left_team:
            # Calculate proximity to ball to determine interceptions
            if np.linalg.norm(player_pos - ball_position) < 0.05:
                reward += self.interception_reward
                self.close_ball_interceptions += 1
  
            # Reward for being deep defensively (close to own goal)
            if player_pos[0] < -0.8:
                reward += self.defensive_position_reward
                self.deep_defensive_actions += 1

        reward_components = {
            "base_score_reward": base_reward,
            "close_ball_interceptions": self.interception_reward,
            "deep_defensive_actions": self.defensive_position_reward
        }

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(np.atleast_1d(value))
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
