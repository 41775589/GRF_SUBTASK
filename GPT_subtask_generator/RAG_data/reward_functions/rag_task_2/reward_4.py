import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive teamwork and strategic positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions_last = None
        self.teamwork_reward_coefficient = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions_last = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "teamwork_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        # Defensive teamwork and positioning rewards
        for rew_index, o in enumerate(observation):
            if 'right_team' in o and 'left_team' in o:
                players = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                ball_position = o['ball']
                # Calculate distances to the ball for each player
                distances = [np.linalg.norm(player[:2] - ball_position[:2]) for player in players]
                # Reward players being close to the ball when their team owns it
                if o['ball_owned_team'] in [-1, 0]:  # if no one or left_team owns the ball
                    teamwork_reward = self.teamwork_reward_coefficient * (1 - np.min(distances) * 0.1) 
                else:
                    teamwork_reward = 0
                    
                components["teamwork_reward"][rew_index] = teamwork_reward
                reward[rew_index] += teamwork_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
