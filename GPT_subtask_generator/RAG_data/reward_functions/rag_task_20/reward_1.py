import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on enhancing offensive strategies by rewarding team coordination and advanced positioning tactics. """

    def __init__(self, env):
        super().__init__(env)
        self.num_zones = 5  # Divide the opponent half into zones for rewards
        self.zone_thresholds = np.linspace(0, 0.5, self.num_zones+1)[1:]  # thresholds for zones
        self.zone_reward = 0.05  # Reward for entering a new zone with the ball
        self.zones_visited = {i: False for i in range(self.num_zones)}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zones_visited = {i: False for i in range(self.num_zones)}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming team 0 is controlled by agent
                ball_x = o['ball'][0]  # x-position of the ball
                active_zone = self.detect_zone(ball_x)
                if active_zone is not None and not self.zones_visited[active_zone]:
                    self.zones_visited[active_zone] = True
                    reward[rew_index] += self.zone_reward
                    components["positioning_reward"][rew_index] += self.zone_reward
        
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
                if action > 0:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
    
    def detect_zone(self, ball_x):
        """Determine the current zone based on ball position."""
        if ball_x > 0:  # Ball must be on the opponent's side
            for i, threshold in enumerate(self.zone_thresholds):
                if ball_x < threshold:
                    return i
            return len(self.zone_thresholds) - 1
        return None
