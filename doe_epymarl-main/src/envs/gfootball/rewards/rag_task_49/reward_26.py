import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for accurate shooting from central field positions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_region_thresholds = (-0.2, 0.2)  # X axis region considered central
        self.critical_distance = 0.2  # Distance threshold from the center of the opponent's goal
        self.accuracy_reward = 3.0  # Reward for shooting from the central region
        self.power_reward = 1.0  # Reward for powerful shots
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),
                      "accuracy_reward": [0.0] * len(reward),
                      "power_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_central_shot = (o['ball'][0] >= self.shooting_region_thresholds[0] and
                               o['ball'][0] <= self.shooting_region_thresholds[1])
            close_to_goal = abs(o['ball'][1] - 1.0) < self.critical_distance
            
            # Give reward for powerful and accurate shots from the center
            if is_central_shot and close_to_goal:
                components["accuracy_reward"][rew_index] = self.accuracy_reward
                reward[rew_index] += components["accuracy_reward"][rew_index]
            
                # Assuming powerful shot by evaluating the speed of the ball (by its direction vector magnitude)
                ball_speed = np.linalg.norm(o['ball_direction'])
                if ball_speed > 0.5:  # threshold for defining powerful shot is set arbitrarily here
                    components["power_reward"][rew_index] = self.power_reward
                    reward[rew_index] += components["power_reward"][rew_index]
        
        return reward, components

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
