import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for long-distance shots from outside the penalty box."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_distance_threshold = 0.6  # Approximate threshold for "long-distance"
        self.goal_y_threshold = 0.044  # Y coordinate threshold for near goal area (penalty box)
        self.score_reward_multiplier = 5.0
        self.shoot_from_distance_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        return self.env.set_state(state)
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_distance_shot_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for index in range(len(reward)):
            obs = observation[index]
            # Check if shooting from distance constraints are fulfilled.
            in_penalty_area = (abs(obs['ball'][0]) >= (1 - self.goal_y_threshold) and
                               abs(obs['ball'][1]) <= self.goal_y_threshold)
            if not in_penalty_area and abs(obs['ball'][0]) >= self.shooting_distance_threshold:
                if 'action' in obs and obs['action'].get('shot', 0) == 1:
                    components['long_distance_shot_reward'][index] = self.shoot_from_distance_reward
                    reward[index] += components['long_distance_shot_reward'][index]
        
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
