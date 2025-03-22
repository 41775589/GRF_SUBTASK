import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the original reward by adding bonuses for successful execution of
    sliding tackles under high-pressure situations. This training is specifically aimed at
    mastering the timing and precision of sliding tackles.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_tackle_bonus = 0.2  # Additional reward for successful sliding under pressure

    def reset(self):
        """
        Resets the environment and any necessary variables.
        """
        return self.env.reset()

    def reward(self, reward):
        """
        Augments the reward based on sliding tackles and pressure conditions:
        - Adds a bonus if a sliding tackle is successfully performed under high-pressure situations.
        
        Parameters:
            reward (list[float]): The list of rewards from the base environment.

        Returns:
            tuple[list[float], dict[str, list[float]]]: A tuple containing the new reward and a dictionary
                                                        of components of the reward.
        """
        observation = self.env.unwrapped.observation()  # Access direct observation from the environment
        components = {"base_score_reward": reward.copy(),  # Keep original reward for component tracking
                      "sliding_tackle_bonus": [0.0]}  # Initialize sliding tackle component

        # Example observation data for checking possession and action success
        # Assume observations include 'sticky_actions' which include an action for sliding tackles
        # Check if the player performed a sliding tackle
        if observation['sticky_actions'][9] == 1:  # Assuming index '9' stands for sliding tackles
            # Define a condition for "high-pressure" situations. For example, near the goal area,
            # or when multiple opponents are close.
            player_pos = observation['left_team'][observation['active']]
            opponents_pos = observation['right_team']
            pressure = False
            for opponent in opponents_pos:
                if np.linalg.norm(np.array(player_pos) - np.array(opponent)) < 0.1:  # Arbitrarily set pressure threshold
                    pressure = True
                    break
            if pressure:
                reward += self.sliding_tackle_bonus
                components['sliding_tackle_bonus'][0] = self.sliding_tackle_bonus

        return reward, components

    def step(self, action):
        """
        Steps through the environment, applying the reward modification through the reward wrapper.
        
        Returns:
            tuple: Observations, rewards, done flag, and info dictionary augmented with reward components.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add each component to 'info' for possible analysis.
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
