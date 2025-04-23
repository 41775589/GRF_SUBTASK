import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on offensive play strategies."""

    def __init__(self, env):
        super().__init__(env)
        # Number of zones towards which progression is measured
        self.num_zones = 5
        # Reward given for entering a new zone with the ball
        self.zone_reward = 0.2
        # Minimum distance on x-axis for the next progression zone
        self.zone_thresholds = np.linspace(0, 1, self.num_zones + 1)[1:]
        # Track the zones reached to give reward for new zones only
        self.reached_zones = [False] * self.num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.reached_zones = [False] * self.num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward
        zone_rewards = [0.0] * len(reward)

        for i, obs in enumerate(observation):
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 1:  # If right team has the ball
                ball_x_position = obs['ball'][0]
                for z in range(self.num_zones):
                    if ball_x_position > self.zone_thresholds[z] and not self.reached_zones[z]:
                        zone_rewards[i] += self.zone_reward
                        self.reached_zones[z] = True

        # Summarize all components
        total_rewards = [base_reward[i] + zone_rewards[i] for i in range(len(reward))]
        reward_components = {
            'base_score_reward': base_reward,
            'zone_rewards': zone_rewards
        }
        return total_rewards, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rewards, components = self.reward(reward)
        info["final_reward"] = sum(rewards)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, rewards, done, info
