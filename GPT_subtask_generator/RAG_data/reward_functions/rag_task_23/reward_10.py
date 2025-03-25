import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward for defensive positioning and coordination in front
    of the penalty box. Encourages agents to cover key defensive zones effectively.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones = [
            # Relative x, y positions of high-pressure key defensive checkpoints around the penalty area
            [0.8, -0.1], [0.8, 0.1], # Right side of penalty box
            [0.9, -0.05], [0.9, 0.05],  # Very close to goal
            [-0.8, -0.1], [-0.8, 0.1], # Left side of penalty box
            [-0.9, -0.05], [-0.9, 0.05]  # Very close to goal
        ]
        self.zone_control_reward = [0] * len(self.defensive_zones)  # track control over defensive zones

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_control_reward = [0] * len(self.defensive_zones)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.zone_control_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zone_control_reward = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_control_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            player_pos = observation[i]['right_team_active']  # assuming our controlled players are always in 'right_team'
            zones_controlled = 0
            for zone_index, zone in enumerate(self.defensive_zones):
                if np.linalg.norm(np.array(player_pos) - np.array(zone)) < 0.1:
                    if self.zone_control_reward[zone_index] == 0:
                        components["defensive_zone_control_reward"][i] += 0.1
                        zones_controlled += 1
                        self.zone_control_reward[zone_index] = 1
            reward[i] += components["defensive_zone_control_reward"][i]
        
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
