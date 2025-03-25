import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for shooting skills focusing on accuracy and power."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shot_power_threshold = 0.5  # Threshold to consider a shot powerful
        self.close_to_goal_threshold = 0.1  # Distance threshold to goal to consider as a good shot position
        self.goal_position = np.array([1, 0])  # Assuming right goal position at x=1, y=0
        self.powerful_shot_reward = 0.3
        self.accuracy_shot_reward = 0.7
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if not observation:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "powerful_shot_reward": [0.0],
                      "accuracy_shot_reward": [0.0]}
        
        ball_position = np.array(observation['ball'][:2])
        ball_direction = np.array(observation['ball_direction'][:2])

        # Check if shot was taken towards the goal with sufficient power
        is_powerful_shot = np.linalg.norm(ball_direction) > self.shot_power_threshold and observation['action'] == 'shot'
        is_accurate_shot = np.linalg.norm(ball_position - self.goal_position) < self.close_to_goal_threshold
        
        if is_powerful_shot:
            reward[0] += self.powerful_shot_reward
            components["powerful_shot_reward"][0] = self.powerful_shot_reward
        
        if is_accurate_shot:
            reward[0] += self.accuracy_shot_reward
            components["accuracy_shot_reward"][0] = self.accuracy_shot_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions info
        for agent_obs in self.env.unwrapped.observation():
            if 'sticky_actions' in agent_obs:
                self.sticky_actions_counter = agent_obs['sticky_actions']
        return observation, reward, done, info
