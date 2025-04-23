import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a creativity and finishing reward based on offensive play and position."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define three zones promoting creativity and finishing :
        # (1) Close range zone near the opponent goal
        # (2) Mid-range zone for creative plays
        # (3) Long-range zone for early stage attacking build-ups
        self.zone_rewards = {
            'close_range': 0.5,
            'mid_range': 0.3,
            'long_range': 0.1
        }
        self.thresholds = {
            'close_range': 0.2,  # Closer than 20% towards the opponent's goal
            'mid_range': 0.5,    # Between 20% and 50%
            'long_range': 0.8     # Between 50% and 80%
        }
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No specific checkpoint state to restore
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, (o, rew) in enumerate(zip(observation, reward)):
            ball_x = o['ball'][0]
            
            if ball_x > 1 - self.thresholds['close_range']:
                zone = 'close_range'
            elif ball_x > 1 - self.thresholds['mid_range']:
                zone = 'mid_range'
            elif ball_x > 1 - self.thresholds['long_range']:
                zone = 'long_range'
            else:
                continue  # No reward allocated if not in an offensive position

            # Heavier reward if the active player's team owns the ball
            if o['ball_owned_team'] == 1:  # Assuming '1' is the team on the right
                components["attack_reward"][rew_index] = self.zone_rewards[zone]
                reward[rew_index] += components["attack_reward"][rew_index]

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
