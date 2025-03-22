import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This wrapper augments the reward in the Google Research Football
       environment to support offensive strategies like shooting, dribbling, and making passes."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward = 0.2
        self.shoot_reward = 0.4
        self.dribble_reward = 0.1
        self.last_ball_position = None

    def reset(self, **kwargs):
        self.last_ball_position = None
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'last_ball_position': self.last_ball_position}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        return from_pickle

    def reward(self, reward):
        current_observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 
                      'pass_reward': [0.0], 'shoot_reward': [0.0], 'dribble_reward': [0.0]}

        # Access observations to inform rewards
        ball_owned_team = current_observation['ball_owned_team']
        game_mode = current_observation['game_mode']
        
        if self.last_ball_position is not None and ball_owned_team != -1:
            dist = np.linalg.norm(current_observation['ball'] - self.last_ball_position)
        
            if dist > 0.2 and game_mode == 5:  # Assuming 5 corresponds to pass
                reward += self.pass_reward
                components['pass_reward'][0] = self.pass_reward
            elif dist > 0.1 and game_mode == 6:  # Assuming 6 corresponds to shoot
                reward += self.shoot_reward
                components['shoot_reward'][0] = self.shoot_reward
        if 'sticky_actions' in current_observation:
            if current_observation['sticky_actions'][9] == 1:  # Assuming index 9 is dribble
                reward += self.dribble_reward
                components['dribble_reward'][0] = self.dribble_reward

        self.last_ball_position = current_observation['ball'].copy()
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add component values to info for transparency
        info['components'] = components
        info['final_reward'] = reward

        return observation, reward, done, info
