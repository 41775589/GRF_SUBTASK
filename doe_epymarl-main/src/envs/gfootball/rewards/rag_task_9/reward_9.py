import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on offensive skills: passing, shooting, and dribbling."""

    def __init__(self, env):
        super().__init__(env)
        # Actions importance scoring coefficients:
        self.short_pass_coeff = 0.1
        self.long_pass_coeff = 0.2
        self.shot_coeff = 0.3
        self.dribble_coeff = 0.1
        self.sprint_coeff = 0.05
        
        # Keep track of sticky actions usage
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter on episode start."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Save additional state if necessary. Not used here."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Load state if necessary. Not used here."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify rewards based on offensive actions executed by agents."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "short_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Increment rewarding based on current agent's actions.
            if o['sticky_actions'][5]:  # Short Pass
                components['short_pass_reward'][rew_index] = self.short_pass_coeff
                reward[rew_index] += components['short_pass_reward'][rew_index]
                
            if o['sticky_actions'][6]:  # Long Pass
                components['long_pass_reward'][rew_index] = self.long_pass_coeff
                reward[rew_index] += components['long_pass_reward'][rew_index]
                
            if o['sticky_actions'][7]:  # Shot
                components['shot_reward'][rew_index] = self.shot_coeff
                reward[rew_index] += components['shot_reward'][rew_index]
                
            if o['sticky_actions'][8]:  # Dribble
                components['dribble_reward'][rew_index] = self.dribble_coeff
                reward[rew_index] += components['dribble_reward'][rew_index]
                
            if o['sticky_actions'][9]:  # Sprint
                components['sprint_reward'][rew_index] = self.sprint_coeff
                reward[rew_index] += components['sprint_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Process environment step, modify rewards, and add reward details to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
