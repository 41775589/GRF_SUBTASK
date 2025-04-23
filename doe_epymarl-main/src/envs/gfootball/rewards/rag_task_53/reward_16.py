import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for ball control under pressure and strategic play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for positional play
        self.ideal_positions = [
            [-0.8, 0],  # Deep left
            [0, 0],     # Deep center
            [0.8, 0],   # Deep right
            [-0.5, 0.3],  # Mid left
            [0, 0.3],     # Mid center
            [0.5, 0.3],   # Mid right
            [-0.3, 0.7], # High left
            [0, 0.7],    # High center
            [0.3, 0.7]   # High right
        ]
        self.position_rewards = np.zeros(len(self.ideal_positions))

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards.fill(0)  # Reset positional rewards for a new episode
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate rewards based on positions and ball control
            if o['ball_owned_team'] == 1:  # If right team has the ball
                ball_position = o['ball'][:2]
                for idx, pos in enumerate(self.ideal_positions):
                    if np.linalg.norm(np.array(ball_position) - np.array(pos)) < 0.1:
                        if self.position_rewards[idx] == 0:
                            components["positional_reward"][rew_index] += 0.05  # Encourage reaching ideal positions
                            self.position_rewards[idx] = 1  # Mark this position as rewarded

            reward[rew_index] += components["positional_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
