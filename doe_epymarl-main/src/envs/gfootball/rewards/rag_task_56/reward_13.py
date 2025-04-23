import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds defensive specialization scores for training in a football environment."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize reward components for shot-stopping and initiating plays (goalkeeper),
        # and tackling and ball-retention (defenders)
        self.goalkeeper_reward = 0.1
        self.defender_reward = 0.1

    def reset(self):
        # Reset sticky actions counter
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Fetch current environment observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_reward": [0.0] * len(reward),
                      "defender_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            roles = o['right_team_roles'] if o['active'] in o['right_team'] else o['left_team_roles']
            
            # Assign specialized rewards based on role and game situation
            if roles[o['active']] == 0:  # Goalkeeper
                if o['ball_owned_player'] == o['active'] and o['game_mode'] in [2, 6]:  # Goal kick or Penalty
                    components["goalkeeper_reward"][rew_index] = self.goalkeeper_reward
                    reward[rew_index] += self.goalkeeper_reward
            
            if roles[o['active']] in [1, 2, 3]:  # Defenders
                if o['ball_owned_player'] == o['active']:
                    components["defender_reward"][rew_index] = self.defender_reward
                    reward[rew_index] += self.defender_reward
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None  # Save relevant internal state
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)  # Load internal state
        # Placeholder for loading state-specific internal data if required
        return from_pickle

    def step(self, action):
        # Handle an environment step and add custom reward components
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Handle possible sticky actions updates
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
