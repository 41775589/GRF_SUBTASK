import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling and stopping under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_threshold = 0.8  # Example threshold for 'pressure'
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_stop_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Identify if player is dribbling
            dribbling = o['sticky_actions'][9] == 1  # Index 9 corresponds to dribbling action

            # Simulate the pressure factor (this should be replaced with a real calculation)
            pressure = self.calculate_pressure(o)

            conditions_to_reward = dribbling and pressure > self.pressure_threshold
            
            # If player stops dribbling under high pressure, reward them
            if conditions_to_reward and self.transitioned_to_stop(o):
                components['dribble_stop_reward'][rew_index] = 0.2
        
        # Update reward for each agent based on individual components
        for i in range(len(reward)):
            reward[i] += components["dribble_stop_reward"][i]

        return reward, components
    
    def calculate_pressure(self, observation):
        """
        Placeholder function to calculate pressure. 
        Actual implementation should quantify pressure based on nearby opponents and game context.
        """
        # Dummy pressure calculation (this should be based on actual nearby opponent proximity and other factors)
        return np.random.rand()

    def transitioned_to_stop(self, observation):
        """
        Check if the player has transitioned from dribbling to stopping.
        This function should check the change in action states between steps, potentially looking at action history.
        """
        # Example check, replace with logic that checks action transition
        return observation['sticky_actions'][9] == 0  # if no dribbling action found

    def step(self, action):
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
