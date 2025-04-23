import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for ball control and strategic play under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_regions = np.linspace(-1, 1, num=9)  # Creates regions across the x-axis of the pitch
        self.control_rewards = np.zeros(len(self.ball_control_regions) - 1)
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.control_rewards.fill(0)  # Reset control rewards for each region
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'control_rewards': self.control_rewards}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.control_rewards = from_pickle['CheckpointRewardWrapper']['control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components
        
        new_rewards = []
        components['strategic_play_reward'] = [0.0] * len(reward)

        for i in range(len(observation)):
            o = observation[i]
            current_reward = reward[i]
            ball_position = o['ball'][0]  # Focus on x-coordinate
            
            # Check if the player's team has control of the ball
            if o['ball_owned_team'] == 1 and self.previous_ball_owner != 1:
                # Check in which region the ball is and if no reward was previously given for that region
                for j in range(len(self.ball_control_regions) - 1):
                    if self.ball_control_regions[j] <= ball_position < self.ball_control_regions[j + 1]:
                        if self.control_rewards[j] == 0:
                            extra_reward = 0.1  # Small reward for maintaining control in this region
                            components['strategic_play_reward'][i] += extra_reward
                            current_reward += extra_reward
                            self.control_rewards[j] = 1  # Mark this region as rewarded
            
            new_rewards.append(current_reward)

            # Update the previous ball owner
            self.previous_ball_owner = o['ball_owned_team']

        return new_rewards, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
