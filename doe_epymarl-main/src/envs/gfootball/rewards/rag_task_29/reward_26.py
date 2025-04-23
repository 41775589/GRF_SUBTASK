import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for shot precision skills specifically for training shots from close range,
    including handling angles and power adjustments to beat the goalkeeper in tight spaces.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define ranges for close range to the goalkeeper (assuming goal area is near x=1)
        self.close_range_threshold = 0.8  # Close range threshold on the x-axis

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Only add precision bonus if it's close range and the player has the ball
            if o['ball'][0] > self.close_range_threshold and o['ball_owned_team'] == 1:
                # Calculate angles to both top and bottom of the goal
                goal_top = 0.044
                goal_bottom = -0.044
                ball_y = o['ball'][1]

                # Calculate angle
                angle_to_top = np.arctan2(goal_top - ball_y, 1 - o['ball'][0])
                angle_to_bottom = np.arctan2(goal_bottom - ball_y, 1 - o['ball'][0])
                
                # Reward player for minimizing angle differential, promotes shooting straight
                angle_diff = abs(angle_to_top - angle_to_bottom)
                components["precision_bonus"][rew_index] = (np.pi/4 - angle_diff) if angle_diff < np.pi/4 else 0

            # Apply rewards
            reward[rew_index] += components["precision_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
