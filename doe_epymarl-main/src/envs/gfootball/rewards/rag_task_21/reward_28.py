import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive play and interceptions in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.5
        self.defensive_position_reward = 0.3
        self.last_ball_possession = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_possession = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_possession'] = self.last_ball_possession
        return state

    def set_state(self, state):
        self.last_ball_possession = state['last_ball_possession']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'interception_reward': [0.0] * len(reward), 'defensive_position_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, obs in enumerate(observation):
            # Reward player for making an interception
            if obs['ball_owned_team'] == 0 and self.last_ball_possession == 1:
                components['interception_reward'][idx] = self.interception_reward
                reward[idx] += components['interception_reward'][idx]
            
            # Evaluate defensive positioning: closer to own goal (x position < 0)
            player_position = obs['left_team'][obs['active']]
            if player_position[0] < -0.5:  # Defensive half
                components['defensive_position_reward'][idx] = self.defensive_position_reward
                reward[idx] += components['defensive_position_reward'][idx]
        
        self.last_ball_possession = observation[0]['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
