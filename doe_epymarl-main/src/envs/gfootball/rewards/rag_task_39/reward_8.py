import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for mastering defensive clearances."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_zones = 5  # Divides the defensive end into 5 zones
        self.zone_rewards = [0.2] * self.clearance_zones  # Reward for clearing from each zone
        self.clearances_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.clearances_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearances_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "clearance_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Only act if this is the defensive half of the field (left team perspective)
            if o['ball'][0] < 0: 
                zone_idx = min(-int(o['ball'][0] / 0.2), self.clearance_zones - 1)  # Calculate zone index
                if zone_idx in self.clearances_collected:
                    # Reward only once per zone per episode
                    continue
                if o['ball_owned_team'] == 0 and o['game_mode'] in [3, 4, 5, 2]:
                    # Check if the game mode is a defensive situation (free kick, corner, throw-in, goal kick)
                    components['clearance_reward'][rew_index] = self.zone_rewards[zone_idx]
                    reward[rew_index] += self.zone_rewards[zone_idx]
                    self.clearances_collected[zone_idx] = True

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
