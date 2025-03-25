import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a targeted reward for practicing long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_length_threshold = 0.3  # Set a threshold for long pass (distance)
        self.accuracy_reward = 0.5  # Reward for accurate long passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate the length of the pass
            if 'ball_direction' in o:
                pass_length = np.linalg.norm(o['ball_direction'][:2])  # Consider x and y components

                # Check if the pass is long enough
                if pass_length > self.pass_length_threshold:
                    # Additional reward for making a long pass
                    components['long_pass_accuracy_reward'][rew_index] = self.accuracy_reward

            # Add the specific components to the reward
            reward[rew_index] = reward[rew_index] + components['long_pass_accuracy_reward'][rew_index]

        return reward, components

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
