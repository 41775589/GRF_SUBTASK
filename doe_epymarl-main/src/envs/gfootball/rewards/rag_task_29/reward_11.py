import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on shot precision and angle adjustments in tight spaces near the goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shot_precision_reward = 0.5  # Extra reward given for precision shots in crucial areas
        self.goal_zone_threshold = 0.2  # Threshold to define 'close range' to the goal on x-axis
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

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

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)  # Error checking

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball'][0]  # X-coordinate of the ball
            if ball_pos > 1 - self.goal_zone_threshold:  # Close to the right goal
                # Calculate the angle of approach
                if o['ball_owned_team'] == 1:  # Right team has the ball
                    angle_to_goal = np.abs(np.arctan2(o['ball'][1], 1 - o['ball'][0]))
                    # Check if ball is within a tight angle from the goal
                    if angle_to_goal < np.pi / 6:
                        components["precision_reward"][rew_index] = self.shot_precision_reward
                        reward[rew_index] += components["precision_reward"][rew_index]
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle
