import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive play and quick counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Dividing the field in 5 vertical zones
        self.zone_reward = 0.05
        self.reset()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_collected = [False] * self._num_zones
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.zone_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zone_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for agent_idx, o in enumerate(observation):
            ball_x = o['ball'][0]
            zone_index = int((ball_x + 1) / 0.4)
            if zone_index < self._num_zones:
                if not self.zone_collected[zone_index]:
                    if o['ball_owned_team'] == 0:  # Defensive play detected
                        reward[agent_idx] += self.zone_reward
                        components["defensive_reward"][agent_idx] = self.zone_reward
                        self.zone_collected[zone_index] = True
                    
                    if o['ball_owned_team'] == 0 and o['game_mode'] == 0 and np.linalg.norm(o['ball_direction']) > 0.1:
                        # Reacting to counterattack opportunity
                        reward[agent_idx] += 2 * self.zone_reward  # extra reward for counterattacking
                        components["defensive_reward"][agent_idx] += 2 * self.zone_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Sum up the final reward values for all agents
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_present in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_present
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
