import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for sprint action focusing on quick positioning to enhance defensive coverage."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_reward = 0.1  # Reward increment for sprint actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the wrapped environment's state."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the wrapped environment's state from a pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Custom reward function that focuses on sprint actions."""
        observation = self.env.unwrapped.observation()
        base_score_reward = np.array(reward, copy=True)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward.tolist()}

        sprint_rewards = np.zeros(len(reward))
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if sprint action is taken
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Index 8 is sprint action
                sprint_rewards[rew_index] = self.sprint_reward
                self.sticky_actions_counter[8] += 1

        # Modify original rewards with sprint rewards
        reward = list(np.array(reward) + sprint_rewards)

        return reward, {'base_score_reward': base_score_reward.tolist(), 'sprint_reward': sprint_rewards.tolist()}

    def step(self, action):
        """Environment step with the modified reward mechanism."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
