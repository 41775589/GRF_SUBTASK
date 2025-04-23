import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive checkpoint reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_efficiency = 0.1
        self._reward_thresholds = [0.8, 0.6, 0.4, 0.2]  # Defensive positions' thresholds
        self._rewards_collected = {}
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._rewards_collected = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            components["defensive_reward"][idx] = 0.0

            if o['game_mode'] != 0:  # Only reward during normal gameplay
                continue

            ball_pos = o['ball']
            player_pos = o['left_team'][o['active']] if o['active_team'] == 0 else o['right_team'][o['active']]
            
            distance_to_ball = np.linalg.norm(player_pos[:2] - ball_pos[:2])  # Consider only x, y
            
            # Evaluate defensive positioning
            for threshold in self._reward_thresholds:
                if distance_to_ball < threshold and idx not in self._rewards_collected:
                    components["defensive_reward"][idx] = self._defensive_efficiency
                    reward[idx] += components["defensive_reward"][idx]
                    self._rewards_collected[idx] = True
                    break
        
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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle
