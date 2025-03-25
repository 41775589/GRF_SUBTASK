import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_thresholds = np.linspace(0.2, 0.8, num=4)  # Thresholds for ball travel to consider a long pass
        self.pass_reward = 0.5  # Reward for achieving each long pass tier
        self.long_pass_counter = 0  # To keep track of how many different pass thresholds were reached
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions

    def reset(self):
        self.long_pass_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['long_pass_counter'] = self.long_pass_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.long_pass_counter = from_pickle['long_pass_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_pass_length = np.linalg.norm(o['ball_direction'][:2])

            # Calculate the accumulated long pass reward based on ball direction length
            for threshold in self.pass_thresholds:
                if current_pass_length > threshold:
                    if self.long_pass_counter < len(self.pass_thresholds):
                        components["long_pass_reward"][rew_index] += self.pass_reward
                        self.long_pass_counter += 1

        # Sum up the base and long pass rewards
        total_reward = [sum(x) for x in zip(reward, components["long_pass_reward"])]
        return total_reward, components

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
