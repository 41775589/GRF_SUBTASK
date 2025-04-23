import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for optimizing shooting angles and timing when close to the goal."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_threshold = 0.1  # Threshold for being 'close' to the goal
        self.angle_reward_coefficient = 0.5  # Coefficient for rewarding better angles
        self.timing_reward_coefficient = 0.3  # Coefficient for rewarding proper timing

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "angle_reward": [0.0] * len(reward),
                      "timing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check if the team has possession near the goal
            if o['ball_owned_team'] != -1 and np.abs(o['ball'][0]) > (1 - self.distance_threshold):
                # Calculate the shoot angle reward
                goal_y = 0.0  # Y-coordinate of the goal center
                ball_y = o['ball'][1]
                angle_penalty = np.abs(ball_y - goal_y)  # smaller angle, higher reward
                angle_reward = (1 - angle_penalty) * self.angle_reward_coefficient
                components["angle_reward"][rew_index] = angle_reward
                
                # Calculate the timing reward based on game mode
                # Higher reward for shooting under pressure situations (e.g., when defenders are close)
                if o['game_mode'] == 0:  # Normal gameplay
                    proximity_penalty = min([np.linalg.norm(player - o['ball'][:2]) 
                                             for player in o['right_team'] if o['ball_owned_team'] == 0] + 
                                            [np.linalg.norm(player - o['ball'][:2]) 
                                             for player in o['left_team'] if o['ball_owned_team'] == 1])
                    timing_reward = max(0, 1 - proximity_penalty) * self.timing_reward_coefficient
                    components["timing_reward"][rew_index] = timing_reward

                reward[rew_index] += components["angle_reward"][rew_index] + components["timing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
