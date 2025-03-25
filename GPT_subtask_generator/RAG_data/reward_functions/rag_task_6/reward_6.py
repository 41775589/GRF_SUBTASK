import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for energy-conserving actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Configure the stamina conservation rewards
        self.sprint_stopped_count = [0, 0]
        self.move_stopped_count = [0, 0]

    def reset(self):
        """Reset the environment state and reward-tracking variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_stopped_count = [0, 0]
        self.move_stopped_count = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add wrapper-specific state information to the pickle."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sprint_stopped_count': self.sprint_stopped_count,
            'move_stopped_count': self.move_stopped_count
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from the pickle information including wrapper-specific data."""
        from_pickle = self.env.set_state(state)
        self.sprint_stopped_count = from_pickle[
            'CheckpointRewardWrapper']['sprint_stopped_count']
        self.move_stopped_count = from_pickle[
            'CheckpointRewardWrapper']['move_stopped_count']
        return from_pickle

    def reward(self, reward):
        """Customize the reward based on energy conservation actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_stopped_reward": [0.0] * len(reward),
                      "move_stopped_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            current_sticky_actions = o['sticky_actions']

            # Reward for stopping sprint: action_sprint index is 8
            if self.sticky_actions_counter[8] == 1 and current_sticky_actions[8] == 0:
                self.sprint_stopped_count[i] += 1
                components["sprint_stopped_reward"][i] += 0.05  # Customizable reward magnitude

            # Reward for stopping movement actions (indices 0 to 7)
            if any(self.sticky_actions_counter[0:8]) and not any(current_sticky_actions[0:8]):
                self.move_stopped_count[i] += 1
                components["move_stopped_reward"][i] += 0.05  # Customizable reward magnitude

            # Update rewards
            reward[i] += components["sprint_stopped_reward"][i] + components["move_stopped_reward"][i]

            # Update sticky actions counters for the next step
            self.sticky_actions_counter = current_sticky_actions.copy()

        return reward, components

    def step(self, action):
        """Step through the environment, applying the customized reward adjustments."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
