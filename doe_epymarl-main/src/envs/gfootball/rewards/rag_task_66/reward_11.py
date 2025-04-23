import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward scheme to promote mastering short pass techniques under pressure."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter = {}
        self.possession_rewards = {}

    def reset(self):
        """Reset the environment and reward counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter.clear()
        self.possession_rewards.clear()
        return self.env.reset()

    def reward(self, reward):
        """Adjust rewards based on short passing effectiveness and ball retention under pressure."""
        observation = self.env.unwrapped.observation()
        reward_adjusted = reward.copy()
        components = {"base_score_reward": reward.copy(), "pass_reward": [0.0] * len(reward)}
        
        for i, obs in enumerate(observation):
            # Short pass rewards for successful passes under defensive pressure
            if ("sticky_actions" in obs and obs["sticky_actions"][9] == 1 and
                obs["ball_owned_team"] == 0 and len(obs["right_team"]) > 0):
                close_opponents = np.any(np.linalg.norm(obs["left_team"][obs["active"]] - obs["right_team"], axis=1) < 0.1)
                if close_opponents:
                    pass_key = (i, obs["ball_owned_team"], obs["ball_owned_player"])
                    if self.pass_completion_counter.get(pass_key, 0) < 3:
                        self.pass_completion_counter[pass_key] = self.pass_completion_counter.get(pass_key, 0) + 1
                        reward_adjusted[i] += 0.2  # reward for completing a pass under pressure
                        components["pass_reward"][i] = 0.2
            
            # Additional reward for retaining possession under high pressure
            if obs["ball_owned_team"] == 0:
                if i not in self.possession_rewards:
                    self.possession_rewards[i] = 0.05  # initial reward for taking possession
                elif self.possession_rewards[i] < 0.3:
                    self.possession_rewards[i] += 0.01  # incrementally reward continued possession
                reward_adjusted[i] += self.possession_rewards[i]
            else:
                self.possession_rewards[i] = 0  # reset on loss of possession

        return reward_adjusted, components

    def step(self, action):
        """Take a step using the specified actions, process rewards, and add metrics to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
