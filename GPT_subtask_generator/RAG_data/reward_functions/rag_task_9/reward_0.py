import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that adds specific rewards for offensive skills in football,
    including passing, shooting, dribbling, and sprinting.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward increments for different successful actions
        self.pass_reward = 0.1
        self.shoot_reward = 0.3
        self.dribble_reward = 0.05
        self.sprint_reward = 0.02

    def reset(self):
        """ Reset the sticky actions counter when the environment is reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """ Include Extra state needed by the wrapper for snapshots. """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore Extra state returned by the wrapper from snapshots. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """ Modifies the reward based on specific offensive actions executed. """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        assert len(reward) == len(observation)

        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            # Check and reward passes (both short and long)
            if o['sticky_actions'][5] == 1 or o['sticky_actions'][6] == 1:   # index for short and long passes in sticky actions
                reward[rew_index] += self.pass_reward
                components['pass_reward'][rew_index] += self.pass_reward
            
            # Check and reward shots
            if o['sticky_actions'][7] == 1:  # index for shot in sticky actions
                reward[rew_index] += self.shoot_reward
                components['shoot_reward'][rew_index] += self.shoot_reward
            
            # Reward for dribble and sprint
            if o['sticky_actions'][8] == 1:  # index for dribble in sticky actions
                reward[rew_index] += self.dribble_reward
                components['dribble_reward'][rew_index] += self.dribble_reward

            if o['sticky_actions'][9] == 1:  # index for sprint in sticky actions
                reward[rew_index] += self.sprint_reward
                components['sprint_reward'][rew_index] += self.sprint_reward
        
        return reward, components

    def step(self, action):
        """ Take action and return results from the environment, combined with the modified reward. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
