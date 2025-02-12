import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for offensive plays in a football game."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_bonus = 0.2
        self.shot_bonus = 1.0
        self.dribble_bonus = 0.3

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Extract the state of the environment for checkpointing."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from a checkpoint."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Calculate specialized rewards based on the type of action performed by agents."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 
                      'pass_reward': [0.0, 0.0], 
                      'shot_reward': [0.0, 0.0], 
                      'dribble_reward': [0.0, 0.0]}
        
        for idx, obs in enumerate(observation):
            # Check if the agent made a pass (indices need to be adjusted according to the sticky_action definitions)
            if 'sticky_actions' in obs and obs['sticky_actions'][7] == 1:
                components['pass_reward'][idx] += self.pass_bonus
                reward[idx] += self.pass_bonus

            # Check if the agent made a shot, assume game mode 3 is a shot at goal
            if obs.get('game_mode', 0) == 3 and obs.get('ball_owned_player', -1) == obs['active']:
                components['shot_reward'][idx] += self.shot_bonus
                reward[idx] += self.shot_bonus

            # Check if the agent used a dribble action (index needs to be checked)
            if 'sticky_actions' in obs and obs['sticky_actions'][6] == 1: # assuming index 6 is dribble
                components['dribble_reward'][idx] += self.dribble_bonus
                reward[idx] += self.dribble_bonus

        return reward, components

    def step(self, action):
        """Take a step in the environment, modify the reward, and return results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, val in components.items():
            info[f"component_{key}"] = sum(val)
        return observation, reward, done, info
