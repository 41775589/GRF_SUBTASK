import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pass_success = 0
        self.defensive_actions = 0
        self.possession_time = 0

    def reset(self):
        self.pass_success = 0
        self.defensive_actions = 0
        self.possession_time = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_success'] = self.pass_success
        to_pickle['defensive_actions'] = self.defensive_actions
        to_pickle['possession_time'] = self.possession_time
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_success = from_pickle.get('pass_success', 0)
        self.defensive_actions = from_pickle.get('defensive_actions', 0)
        self.possession_time = from_pickle.get('possession_time', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()[0]  # Assuming single agent context

        base_score_reward = reward[0]
        components = {
            "base_score_reward": [base_score_reward],
            "pass_accuracy_reward": [0.0],
            "defensive_actions_reward": [0.0],
            "possession_time_reward": [0.0]
        }
        
        # Evaluate condition for successful passes
        if 'ball_owned_team' in observation and observation['ball_owned_team'] == 0:
            self.pass_success += 1
            components["pass_accuracy_reward"][0] = self.pass_success * 0.1
        
        # Handling defensive actions based on game mode
        if 'game_mode' in observation and observation['game_mode'] in [3, 4]:  # Assuming these modes are defensive
            self.defensive_actions += 1
            components["defensive_actions_reward"][0] = self.defensive_actions * 0.2
        
        # Managing possession time enhancement
        if observation.get('ball_owned_player') is not None and observation.get('ball_owned_player') == observation.get('active'):
            self.possession_time += 1
            components["possession_time_reward"][0] = self.possession_time * 0.05

        # Calculating the total reward
        total_reward = sum([sum(value) for key, value in components.items()])
        return [total_reward], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = reward[0]
        
        for key, value in components.items():
            info[f'component_{key}'] = value[0]
        
        return observation, reward, done, info
