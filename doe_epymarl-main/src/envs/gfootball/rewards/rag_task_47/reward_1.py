import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized dense reward for mastering sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones = {
            # Define zones based on y-coordinates closer to the defending goal
            'close_defense': (-1, -0.2),
            'mid_defense': (-0.2, -0.5),
            'far_defense': (-0.5, -1)
        }
        self.reward_for_tackle = 0.3  # Reward for successful tackle in defensive zone

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
        # Extract the observations from the environment
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            # Determine if a tackle occurred
            if obs['sticky_actions'][9] and obs['ball_owned_team'] != 0:
                # Check player is in one of the defensive zones
                player_y = obs['left_team'][obs['designated']][1] if obs['active'] < obs['left_team'].shape[0] else obs['right_team'][obs['designated']][1]
                
                for zone, y_range in self.defensive_zones.items():
                    if y_range[0] <= player_y <= y_range[1]:
                        components["tackle_reward"][rew_index] = self.reward_for_tackle
                        reward[rew_index] += components["tackle_reward"][rew_index]
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
