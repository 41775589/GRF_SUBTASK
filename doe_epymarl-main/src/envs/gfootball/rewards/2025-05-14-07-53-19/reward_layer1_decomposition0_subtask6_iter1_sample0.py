import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically focuses on enhancing dribbling skills and managing stop-dribble tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._stop_dribble_reward = 0.4
        self._dribble_reward = 0.1
        self._pressure_threshold = 0.05  # Simulated 'pressure' metric threshold
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Obtain the environment's observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0],
                      "stop_dribble_reward": [0.0]}

        if observation is None:
            return reward, components
        
        o = observation[0]  # Single agent scenario
        
        # Check if the player has the ball and compute distance to closest opponent
        owned_by_active = (o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0)
        player_pos = o['left_team'][o['active']]
        opponents_pos = o['right_team']
        closest_opponent_dist = np.min(np.linalg.norm(player_pos - opponents_pos, axis=1))

        # Reward for dribbling while maintaining ball possession under pressure
        if o['sticky_actions'][9] == 1 and owned_by_active and closest_opponent_dist < self._pressure_threshold:
            reward[0] += self._dribble_reward  # Add dribbling reward
            components["dribble_reward"][0] = self._dribble_reward

        # Reward for effectively stopping dribble while holding the ball under opponent pressure
        if o['sticky_actions'][9] == 0 and owned_by_active and closest_opponent_dist < self._pressure_threshold:
            reward[0] += self._stop_dribble_reward  # Add stop dribble reward
            components["stop_dribble_reward"][0] = self._stop_dribble_reward

        return [reward[0]], components

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
