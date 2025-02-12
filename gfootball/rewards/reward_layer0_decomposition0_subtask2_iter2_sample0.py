import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        # Forward the state retrieval to the underlying environment
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the state to the underlying environment
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        o = observation[0]  # Assume single-agent environment for simplicity
        
        # Initialize reward components dict
        components = {
            "base_score_reward": reward.copy(),  # base reward before modification
            "possession_reward": [0.0],
            "defensive_action_reward": [0.0],
            "passing_quality_reward": [0.0]
        }

        # Reward for possession of the ball
        if o['ball_owned_team'] == 0:  # Assuming 0 is the index for the controlled team
            components["possession_reward"][0] = 0.1
        
        # Reward for defensive actions based on game modes that indicate defensive situations (e.g., corner, free kick against)
        if o['game_mode'] in [2, 3, 4]:  # Hypothetical indices for game modes that need defensive action
            components["defensive_action_reward"][0] = 0.2
        
        # Reward for successful long and high passes
        if 'sticky_actions' in o:
            if o['sticky_actions'][6] or o['sticky_actions'][7]:  # indices for LongPass, HighPass
                components["passing_quality_reward"][0] = 0.3

        # Summing up the reward components
        total_reward = []
        for key, value in components.items():
            if key == "base_score_reward":
                continue  # The base score reward is included by default
            total_reward.append(sum(value))
        
        final_reward = sum(total_reward) + reward[0]
        
        return [final_reward], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add final reward and components for detailed tracking
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
