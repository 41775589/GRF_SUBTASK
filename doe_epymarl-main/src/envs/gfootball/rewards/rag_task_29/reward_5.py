import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for precision in tight space shooting near the goal."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.precision_zones = [
            {'x': 0.8, 'y': 0.2, 'radius': 0.1, 'reward': 0.2},
            {'x': 0.8, 'y': -0.2, 'radius': 0.1, 'reward': 0.2},
            {'x': 0.9, 'y': 0.1, 'radius': 0.05, 'reward': 0.3},
            {'x': 0.9, 'y': -0.1, 'radius': 0.05, 'reward': 0.3},
            {'x': 1.0, 'y': 0.0, 'radius': 0.03, 'reward': 0.5}
        ]
        self.visited_precision_zones = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.visited_precision_zones = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'visited_precision_zones': self.visited_precision_zones}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.visited_precision_zones = from_pickle['CheckpointRewardWrapper']['visited_precision_zones']
        return from_pickle
    
    def reward(self, reward):
        """
        Adds reward based on precision shooting in tight spaces close to the goal.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy()}
        for zone in self.precision_zones:
            ball_pos = observation['ball'][:2]
            dist_to_zone = np.linalg.norm(np.array([zone['x'], zone['y']]) - np.array(ball_pos))
            if dist_to_zone <= zone['radius'] and zone not in self.visited_precision_zones:
                reward += zone['reward']
                self.visited_precision_zones.append(zone)
                break

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
