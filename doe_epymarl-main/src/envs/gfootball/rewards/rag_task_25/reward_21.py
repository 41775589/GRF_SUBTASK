import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for dribbling and sprint actions."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize the counter for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_reward = 0.05
        self.sprint_reward = 0.1

    def reset(self):
        """Reset the counter and environment states."""
        # Reset sticky action counters
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get state information for recreation."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from the given pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        """Reward for dribbling and sprinting actions."""
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Identifying Dribbling action and Sprinting action
            if 'sticky_actions' in o:
                action_sprint = int(o['sticky_actions'][8])
                action_dribble = int(o['sticky_actions'][9])
                
                if action_dribble:
                    components["dribbling_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]
                
                if action_sprint:
                    components["sprinting_reward"][rew_index] = self.sprint_reward
                    reward[rew_index] += components["sprinting_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Step function to execute environment step and modify reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Include modified rewards in info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
