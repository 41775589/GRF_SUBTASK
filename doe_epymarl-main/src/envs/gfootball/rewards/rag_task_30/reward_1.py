import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering strategic positioning and quick transitions from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for important actions
        self.defensive_positions_reached_counter = np.zeros(10, dtype=int)
        self.transition_speed_bonus = np.zeros(10, dtype=float)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.defensive_positions_reached_counter.fill(0)
        self.transition_speed_bonus.fill(0.0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward strategic positioning
            if o['active'] in (0, 1):  # Only rewarding defensive roles like GK, CB
                defensive_pos = o['left_team'][o['active']]
                if defensive_pos[0] < -0.5:  # Back on the left side
                    components["positioning_reward"][rew_index] = 1.0
                    reward[rew_index] += components["positioning_reward"][rew_index]

            # Reward for faster transition
            if (o['game_mode'] == 3 or o['game_mode'] == 4) and o['ball_owned_team'] == 0:
                # Assume game mode 3/4 are related to defensive states like free kicks or corners
                time_to_cross_half = np.abs(o['ball'][0]) # Time normalized by distance
                self.transition_speed_bonus[rew_index] = max(0, (0.5 - time_to_cross_half) * 2)
                components["transition_reward"][rew_index] = self.transition_speed_bonus[rew_index]
                reward[rew_index] += components["transition_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Aggregate reward for all steps
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Tracking sticky actions as part of information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
