import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for ball control and strategic space exploitation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_control_rewards = np.zeros(3, dtype=float)
        self._ball_control_threshold = 0.1
        self._space_exploit_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_control_rewards = np.zeros(3, dtype=float)
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
                      "control_under_pressure": [0.0] * len(reward),
                      "space_exploitation": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            
            # Reward for maintaining control of the ball under pressure
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                distance_to_nearest_opponent = np.min(np.linalg.norm(
                    o['right_team'][o['active']] - o['left_team'], axis=1))
                if distance_to_nearest_opponent < self._ball_control_threshold:
                    components["control_under_pressure"][i] = self._space_exploit_reward
                    reward[i] += components["control_under_pressure"][i]
            
            # Reward for strategic space exploitation
            own_team_positions = o['right_team']
            opponent_positions = o['left_team']
            for pos in own_team_positions:
                if np.any(np.linalg.norm(pos - opponent_positions, axis=1) > 0.3):  # Checking for open spaces
                    components["space_exploitation"][i] += self._space_exploit_reward
                    break
            reward[i] += components["space_exploitation"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Count sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
