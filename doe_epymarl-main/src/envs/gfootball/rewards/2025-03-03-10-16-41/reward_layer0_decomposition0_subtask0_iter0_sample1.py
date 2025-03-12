import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for offensive actions."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        
        if observations:
            for rew_index, o in enumerate(observations):
                # Check if the player is in possession of the ball
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    # Add dense checkpoint reward for offensive actions like passing and shooting
                    if 'Shot' in o['sticky_actions'] or 'Short Pass' in o['sticky_actions'] or 'High Pass' in o['sticky_actions'] or 'Long Pass' in o['sticky_actions'] or 'Dribble' in o['sticky_actions'] or 'Sprint' in o['sticky_actions']:
                        components["checkpoint_reward"][rew_index] = 0.1
                        reward[rew_index] = 1 * components["base_score_reward"][rew_index] + components["checkpoint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
