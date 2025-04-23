import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances reward signals for dribbling and sprint actions."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_reward = 0.05
        self.sprint_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and resets tracking for dribble and sprint actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Provides state information for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state from serialized data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Enhances the reward based on dribbling and sprinting actions.
        
        Args:
            reward (list[float]): The base reward from the environment.

        Returns:
            tuple[list[float], dict[str, list[float]]]: The adjusted rewards and components breakdown.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Encourage dribbling and using sprint strategically
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward dribbling when it contributes to maintaining control
            if o['sticky_actions'][9]:  # dribble action index
                components["dribble_reward"][rew_index] = self.dribble_reward
                
            # Reward sprinting when moving aggressively forward
            if o['sticky_actions'][8]:  # sprint action index
                # Check if the direction is forward towards the opponent's goal
                if o['ball'][0] > 0.5:  # Assuming the team is on the right side if ball x > 0.5
                    components["sprint_reward"][rew_index] = self.sprint_reward
            
            reward[rew_index] += (components["dribble_reward"][rew_index] +
                                  components["sprint_reward"][rew_index])

        return reward, components

    def step(self, action):
        """Performs the action in the environment and adjusts rewards with detailed breakdown."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
