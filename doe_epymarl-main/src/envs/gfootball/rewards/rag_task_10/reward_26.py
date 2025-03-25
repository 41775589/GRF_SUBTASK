import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive actions preventing scoring."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Initialize the custom components of reward
        components = {"base_score_reward": reward.copy(), "defensive_action_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        
        # Apply the defensive reward logic
        for i in range(len(reward)):
            player_obs = observation[i]
            active_player = player_obs['active']
            team_actions = player_obs['sticky_actions']
            
            # Check for defensive actions
            slide_tackle = team_actions[10]  # Example index for slide tackle
            intercept = team_actions[11]  # Example index for intercept
            
            if slide_tackle or intercept:
                components["defensive_action_reward"][i] = self.defensive_actions_reward
                
            # Update the reward for this player
            reward[i] += components["defensive_action_reward"][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
