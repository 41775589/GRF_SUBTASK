import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes midfield control and paced game transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and counters for the new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Returns the state of the environment to be pickled."""
        # Here custom states can be added to the pickle
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the environment state from an unpickled state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modifies the reward based on midfield control and transition speed."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, dtype=float)}
        
        for i, obs in enumerate(observation):
            # Component for maintaining control in the midfield
            midfield_control = np.any((obs['left_team'][:, 0] > -0.2) & (obs['left_team'][:, 0] < 0.2))
            components.setdefault("midfield_control", []).append(0.1 if midfield_control else 0.0)

            # Component for managing game pace (using player tiredness)
            average_tiredness = np.mean(obs['left_team_tired_factor'])
            components.setdefault("pace_management", []).append(-0.05 if average_tiredness > 0.1 else 0.05)

            # Updating the reward with new components:
            reward[i] += components["midfield_control"][-1] + components["pace_management"][-1]

        return reward, components

    def step(self, action):
        """Steps through the environment, adjusts reward, and logs extra info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
