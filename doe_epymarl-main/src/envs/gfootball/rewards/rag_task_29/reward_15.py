import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting precision reward for close range shots."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_threshold = 0.1    # Threshold to consider a close range
        self.shot_taken = False
        self.goal_scored = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_taken = False
        self.goal_scored = False
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]  # Only x, y coordinates
            right_goal = [1, 0]  # Right-side goal center

            # Determine if a shot is taken within the threshold distance to goal
            if (euclidean(ball_pos, right_goal) <= self.shooting_threshold and 
                    o['ball_owned_team'] == 1):  # Check ball owned by right team
                self.shot_taken = True
            
            # Check if a goal is scored after recent shot
            if self.shot_taken and reward[rew_index] > 0:
                self.goal_scored = True
            
            # Provide additional reward for precision in close range shot
            if self.goal_scored:
                components["shooting_reward"][rew_index] = 1.0  # Same as scoring a goal
                reward[rew_index] += components["shooting_reward"][rew_index]
                # Reset conditions after rewarding
                self.shot_taken = False
                self.goal_scored = False

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add reward parts to info for debug purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions counter and update with current actions
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
