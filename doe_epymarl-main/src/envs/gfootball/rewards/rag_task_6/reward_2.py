import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that reinforces efficient usage of sprint and movement actions to conserve energy.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # This counts how many steps an agent has been using specific sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward increment when stopping sprint/moving efficiently
        self.sprint_movement_efficiency_reward = 0.05
        # Counts when the agent effectively stops movement/sprint
        self.stop_moving_efficiency_count = 0
        
    def reset(self):
        """
        Reset the environment and reward counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_moving_efficiency_count = 0
        return self.env.reset()
        
    def get_state(self, to_pickle):
        """
        Get the environment state and add wrapper-specific state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['stop_moving_efficiency_count'] = self.stop_moving_efficiency_count
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        """
        Set the state of the environment along with wrapper-specific state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.stop_moving_efficiency_count = from_pickle['stop_moving_efficiency_count']
        return from_pickle
    
    def reward(self, reward):
        """
        Augment the existing reward by adding bonuses for stopping sprint and moving actions efficiently.
        """
        # Access the observation directly
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": np.array(reward).copy(),
            "efficient_stop_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Check proactive stopping of actions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']
            
            # Check relevant actions (sprint is index 8 and movement indexes are from 0-7)
            moving_action_indices = list(range(8))
            sprint_index = 8
            currently_moving = any(sticky_actions[i] for i in moving_action_indices)
            currently_sprinting = sticky_actions[sprint_index] == 1
            
            # Rewarding the stop of excessive movement or sprint
            if self.sticky_actions_counter[sprint_index] > 10 and not currently_sprinting:
                components["efficient_stop_reward"][rew_index] += self.sprint_movement_efficiency_reward
                reward[rew_index] += components["efficient_stop_reward"][rew_index]
                
            if self.sticky_actions_counter[sprint_index] > 0 and not currently_sprinting:
                self.stop_moving_efficiency_count += 1
                
            # Update counters
            self.sticky_actions_counter = np.array(sticky_actions, dtype=int)

        return reward, components

    def step(self, action):
        """
        Take a step using the given actions, process the reward components, and add diagnostic information to the `info`.
        """
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
