import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on offensive football strategies: shooting, dribbling, and passing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize any necessary variables here
        self.pass_reward = 0.5
        self.shot_reward = 1.0
        self.dribble_reward = 0.3
        self.goal_reward = 2.0
    
    def reset(self):
        # Reset the environment and any necessary variables
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store state-specific information
        to_pickle['CheckpointRewardWrapper'] = vars(self)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore state-specific information
        vars(self).update(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        # Additional reward calculations
        reward_components = {"base_score_reward": reward.copy()}
        observation = self.env.unwrapped.observation()

        # Initialize component rewards per agent
        reward_components["pass_reward"] = [0.0] * len(reward)
        reward_components["shot_reward"] = [0.0] * len(reward)
        reward_components["dribble_reward"] = [0.0] * len(reward)

        for i, o in enumerate(observation):
            # If the team scores, add a goal reward
            if o['score'][1] > o['score'][0]:  # Assuming the agent is playing for righ_team
                reward_components["goal_reward"] = self.goal_reward
            
            # Player has ball and shoots towards goal (check for shoot direction and position near goal)
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0.5:  # Ball in opponent's half
                reward[i] += self.shot_reward
                reward_components["shot_reward"][i] += self.shot_reward
            
            # Player dribbles effectively
            if o['sticky_actions'][8] > 0:  # Check if dribble action is set
                reward[i] += self.dribble_reward
                reward_components["dribble_reward"][i] += self.dribble_reward
            
            # For completing successful passes
            if 'action' in o and o['action'] == 'action_short_pass':  # Check if a pass action happened
                reward[i] += self.pass_reward
                reward_components["pass_reward"][i] += self.pass_reward
        
        return reward, reward_components

    def step(self, action):
        # Execute a step using the underlying environment
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the custom reward function
        reward, components = self.reward(reward)

        # Add the processed rewards to info for analysis and debugging
        for key in components:
            info['reward_component_{}'.format(key)] = sum(components[key])

        return observation, reward, done, info
