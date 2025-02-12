import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper designed specifically for a midfielder/advance defender training.
    Boosts the reward for successful high passes, long passes, dribbling under pressure,
    and effective midfield defensive and transitional plays.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_evaluation = 0
        self.dribbling_effectiveness = 0
        self.transitioning_effectiveness = 0

    def reset(self):
        # Resetting specific metrics at the start of each episode
        self.pass_evaluation = 0
        self.dribbling_effectiveness = 0
        self.transitioning_effectiveness = 0
        return self.env.reset()
        
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state_data = {
            'pass_evaluation': self.pass_evaluation,
            'dribbling_effectiveness': self.dribbling_effectiveness,
            'transitioning_effectiveness': self.transitioning_effectiveness
        }
        state.update(state_data)
        return state

    def set_state(self, state):
        state_data = self.env.set_state(state)
        self.pass_evaluation = state_data.get('pass_evaluation', 0)
        self.dribbling_effectiveness = state_data.get('dribbling_effectiveness', 0)
        self.transitioning_effectiveness = state_data.get('transitioning_effectiveness', 0)
        return state_data

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = reward.copy()
        
        components = {
            "base_score_reward": reward.copy(),
            "pass_evaluation_reward": [0.0],
            "dribbling_effectiveness_reward": [0.0],
            "transitioning_effectiveness_reward": [0.0]
        }
        
        if observation is None or not observation:
            return reward, components

        o = observation[0]  # Assuming single-agent
        # Evaluate passing - focus on precise high and long passes
        if o['game_mode'] in [4, 5]:  # Assuming these modes are for high and long passes
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:  # Possession by agent
                self.pass_evaluation += 1
                components["pass_evaluation_reward"][0] = 0.1  # Increment reward

        # Dribbling under pressure
        if o['sticky_actions'][9]:  # Assuming index 9 corresponds to dribble action
            self.dribbling_effectiveness += 1
            components["dribbling_effectiveness_reward"][0] = 0.1

        # Effective transitioning in defense to offense and vice versa
        if o['ball_direction'][0] > 0 or o['ball_direction'][1] > 0:  # If there is a significant ball movement positive for our team
            self.transitioning_effectiveness += 1
            components["transitioning_effectiveness_reward"][0] = 0.2

        # Summing all the rewards up
        total_reward = sum(modified_reward) + sum(components[k][0] for k in components if k != 'base_score_reward')
        modified_reward = [total_reward]
        
        return modified_reward, components

    def step(self, action):
        # Standard step function should not be modified according to the problem statement
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
