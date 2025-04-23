import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on transition skills from defense to attack,
    specifically emphasizing Short Pass, Long Pass, and Dribble under pressure.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define custom reward components with respective weights
        self.short_pass_bonus = 0.05
        self.long_pass_bonus = 0.1
        self.dribble_bonus = 0.075


    def reset(self):
        """
        Reset the reward wrapper and the environment.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current state of the environment including the wrapper status.
        """
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of both the environment and the reward wrapper from saved state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on possession control, short pass, long pass, and dribbling effectiveness.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(), 
                      "short_pass_bonus": 0.0,
                      "long_pass_bonus": 0.0,
                      "dribble_bonus": 0.0}
        
        for idx, agent_obs in enumerate(observation):
            if agent_obs['sticky_actions'][8] == 1:  # Dribble action active
                reward[idx] += self.dribble_bonus
                components['dribble_bonus'] += self.dribble_bonus
            
            if agent_obs['sticky_actions'][9] == 1:  # Short pass action active
                reward[idx] += self.short_pass_bonus
                components['short_pass_bonus'] += self.short_pass_bonus
            
            if agent_obs['sticky_actions'][7] == 1:  # Long pass action active
                reward[idx] += self.long_pass_bonus
                components['long_pass_bonus'] += self.long_pass_bonus

        return reward, components
    
    def step(self, action):
        """
        Take an action using the underlying environment and apply the reward modification.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track sticky actions across steps
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
