import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds defensive skill rewards focusing on interception, marking, and blocking opponent's attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initializing count of defensive actions which are rewarded.
        self.interceptions = 0
        self.blocks = 0
        self.markings = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define constants for additional rewards
        self._interception_reward = 0.5
        self._block_reward = 0.3
        self._marking_reward = 0.2

    def reset(self):
        """Reset the reward counters and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.blocks = 0
        self.markings = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper and environment."""
        to_pickle['interceptions'] = self.interceptions
        to_pickle['blocks'] = self.blocks
        to_pickle['markings'] = self.markings
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper and environment."""
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle.get('interceptions', 0)
        self.blocks = from_pickle.get('blocks', 0)
        self.markings = from_pickle.get('markings', 0)
        return from_pickle

    def reward(self, reward):
        """Enhance reward based on defensive actions performed during the step."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": 0.0,
                      "block_reward": 0.0,
                      "marking_reward": 0.0}
        
        if observation is None:
            return reward, components

        # Check for defensive actions using sticky actions and ball ownership
        # Here we assume defensive actions indices correspond to specific actions.
        for agent_index in range(len(reward)):
            agent_obs = observation[agent_index]
            if agent_obs['sticky_actions'][8] == 1 and agent_obs['ball_owned_team'] == 1:  # Example index for interception
                self.interceptions += 1
                components['interception_reward'] += self._interception_reward
            
            if agent_obs['sticky_actions'][9] == 1 and agent_obs['ball_owned_team'] == 1:  # Example index for block
                self.blocks += 1
                components['block_reward'] += self._block_reward
            
            if agent_obs['sticky_actions'][7] == 1 and agent_obs['ball_owned_team'] == 0:  # Example index for marking
                self.markings += 1
                components['marking_reward'] += self._marking_reward
            
            # Aggregate rewards
            reward[agent_index] += (components['interception_reward'] +
                                    components['block_reward'] +
                                    components['marking_reward'])
        
        return reward, components

    def step(self, action):
        """Evaluates the environment’s step function and returns enhanced information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
