import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for the 'Stop-Sprint and Stop-Moving' technique in defensive actions.
    The task focuses on abrupt stopping to handle quick direction changes defensively.
    """

    def __init__(self, env):
        super().__init__(env)
        self.stop_sprint_reward = 0.05
        self.stop_movement_reward = 0.03
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Calculate the reward components for each agent.
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
            
        components = {"base_score_reward": reward,
                      "stop_sprint_reward": [0.0, 0.0],
                      "stop_movement_reward": [0.0, 0.0]}
        
        # Loop over agents' observations.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            sticky_actions = o['sticky_actions']
            action_sprint = sticky_actions[8]  # Sprint action index.
            is_stopped = all([not action for action in sticky_actions[:8]])  # Check if agent is not moving.
            
            # Check for sprint stopping.
            if self.sticky_actions_counter[8] and not action_sprint:
                components["stop_sprint_reward"][rew_index] = self.stop_sprint_reward
                reward[rew_index] += components["stop_sprint_reward"][rew_index]
            
            # Check for movement stopping.
            if is_stopped:
                components["stop_movement_reward"][rew_index] = self.stop_movement_reward
                reward[rew_index] += components["stop_movement_reward"][rew_index]
            
            self.sticky_actions_counter[8] = action_sprint  # Update sprint action state.
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Include components in info dictionary for detailed debugging.
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Count sticky actions.
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info

    def get_state(self, to_pickle):
        # Custom method for getting state with wrapper's details
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Custom method for setting state with wrapper's details
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
