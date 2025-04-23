import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that introduces defensive coordination and transition from 
    defense to attack through secure ball distribution rewards. """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize reward attributes
        self._possession_change_reward = 0.2
        self._defensive_efficiency_reward = 0.1
        self.previous_ball_owner_team = None
        self.possession_changes = 0

    def reset(self):
        """ Reset the environment and reward related parameters. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner_team = None
        self.possession_changes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Include reward-related state items for proper continuation after a pause. """
        to_pickle['previous_ball_owner_team'] = self.previous_ball_owner_team
        to_pickle['possession_changes'] = self.possession_changes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restores the game state including reward-related parameters. """
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner_team = from_pickle['previous_ball_owner_team']
        self.possession_changes = from_pickle['possession_changes']
        return from_pickle

    def reward(self, reward):
        """ Modifies the reward based on defensive efficiency and switches in possession. """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "possession_change_reward": [0.0] * len(reward),
            "defensive_efficiency_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Reward players for changing possession effectively
            current_ball_owner_team = o.get('ball_owned_team')
            if current_ball_owner_team != -1 and current_ball_owner_team != self.previous_ball_owner_team:
                # Changing possession effectively
                self.possession_changes += 1
                components["possession_change_reward"][rew_index] = self._possession_change_reward
                reward[rew_index] += components["possession_change_reward"][rew_index]

            # Reward for defending efficiently by maintaining ball possession under pressure
            if current_ball_owner_team == o['left_team'] and o['game_mode'] == 0 and o['ball_owned_team'] == o['designated']:
                components["defensive_efficiency_reward"][rew_index] = self._defensive_efficiency_reward
                reward[rew_index] += components["defensive_efficiency_reward"][rew_index]

            self.previous_ball_owner_team = current_ball_owner_team
            
        return reward, components

    def step(self, action):
        """ Perform a step action, capture the reward, and append informational components. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)  # reset sticky actions counter
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
