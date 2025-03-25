import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a stamina conservation reward based on proper usage of Stop-Sprint and Stop-Moving actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To track the use of sticky actions
        self.sprint_usage_counter = np.zeros(2, dtype=int)  # To count sprint actions
        self.stop_usage_counter = np.zeros(2, dtype=int)  # To count stop actions
        
    def reset(self):
        """Reset the sticky actions and stop/sprint counters."""
        self.sprint_usage_counter.fill(0)
        self.stop_usage_counter.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Save the current state of the wrapper along with the environment state."""
        to_pickle['sprint_usage'] = self.sprint_usage_counter.tolist()
        to_pickle['stop_usage'] = self.stop_usage_counter.tolist()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Restore the state of the wrapper along with the environment state."""
        from_pickle = self.env.set_state(state)
        self.sprint_usage_counter = np.array(from_pickle['sprint_usage'])
        self.stop_usage_counter = np.array(from_pickle['stop_usage'])
        return from_pickle
    
    def reward(self, reward):
        """Modify the rewards based on energy conservation strategies."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'sprint_usage_reward': [0.0] * len(reward),
                      'stop_usage_reward': [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check sprint usage
            if o['sticky_actions'][8]:  # Sprint action index
                self.sprint_usage_counter[rew_index] += 1
            # Reward negative for excessive sprint without stopping
            components['sprint_usage_reward'][rew_index] = -0.1 * self.sprint_usage_counter[rew_index] if self.sprint_usage_counter[rew_index] > 10 else 0
            
            # Check for moving stop or sprint stop
            if not o['sticky_actions'][5] and not o['sticky_actions'][8]:  # Neither moving nor sprinting
                self.stop_usage_counter[rew_index] += 1
                # Reward positively for good stop usage to manage stamina
                components['stop_usage_reward'][rew_index] = 0.05 * self.stop_usage_counter[rew_index]
            
            # Combine base reward with auxiliary rewards
            reward[rew_index] += (components['sprint_usage_reward'][rew_index] + 
                                  components['stop_usage_reward'][rew_index])
        
        return reward, components
    
    def step(self, action):
        """Execute a step with the wrapped environment and modify rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update info about sticky actions for debugging
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
