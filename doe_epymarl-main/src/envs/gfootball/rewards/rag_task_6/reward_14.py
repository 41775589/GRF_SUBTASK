import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds functionality to track efficient functionality usage."""
    
    def __init__(self, env):
        # Intializing the gym RewardWrapper.
        super().__init__(env)
        
        # Tracking the balance between using sprint and non-sprint actions.
        self.action_usage = {
            'sprint': 0,
            'stop_sprint': 0,
            'sprint_penalty': -0.01,
            'non_sprint_bonus': 0.01
        }
        
        # Track the number of steps to evaluate sustained action usage over time.
        self.steps_since_last_action_change = 0
        
        # Initialize a counter for sticky actions to track continuity and change.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.steps_since_last_action_change = 0
        return self.env.reset()

    def get_state(self, to_pickle={}):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter'] 
        return from_pickle

    def reward(self, reward):
        # Obtain current observation to determine actions taken.
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "efficiency_reward": [0.0] * len(reward)
        }
        
        # Iterate over each agent's observation
        for i, o in enumerate(observation):
            current_actions = o['sticky_actions']
            
            is_sprinting = current_actions[8] == 1  # index 8 corresponds to action_sprint
            has_stopped_sprinting = self.sticky_actions_counter[8] > 0 and current_actions[8] == 0
            
            # Reward conservation of energy: Using sprint effectively without wasting it.
            if is_sprinting:
                self.action_usage['sprint'] += 1
            if has_stopped_sprinting:
                self.action_usage['stop_sprint'] += 1
                        
            # Apply sprint penalty and non-sprint bonus based on action statistics.
            if is_sprinting and self.steps_since_last_action_change > 10:
                components['efficiency_reward'][i] += self.action_usage['sprint_penalty']
            if not is_sprinting:
                components['efficiency_reward'][i] += self.action_usage['non_sprint_bonus']
            
            # Update the total reward with the calculated efficiency rewards.
            reward[i] += components['efficiency_reward'][i]
            
            # Update sticky actions tracker for next step comparisons.
            self.sticky_actions_counter = current_actions

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding detailed breakdown of rewards into 'info'
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
