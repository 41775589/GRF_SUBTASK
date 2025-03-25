import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward tailored to training a 'sweeper' player."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Constants for the reward function
        self.position_threshold = 0.5  # Threshold in x-axis to consider the player in the defensive zone
        self.clearance_reward = 0.3  # Reward for clearing the ball from the defensive zone
        self.tackle_reward = 0.2  # Reward for successful tackles as the last man
        self.recovery_speed = 0.1  # Reward for quick recovery to the defensive position

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on the sweeper's activities."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0],
                      "tackle_reward": [0.0],
                      "recovery_reward": [0.0]}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check if the player is in the defensive zone and has cleared the ball
            if o['ball'][0] < self.position_threshold and o['ball_owned_team'] == -1:
                components['clearance_reward'][rew_index] = self.clearance_reward

            # Check if it was a successful tackle as the last man
            if o['game_mode'] == 3 and 'last_man_tackle' in o:  # Simulated game mode and action tag
                components['tackle_reward'][rew_index] = self.tackle_reward

            # Speed of recovery to the back
            if o['active'] == o['designated'] and o['ball'][0] > 0.5:  # Active and positioned forward
                components['recovery_reward'][rew_index] = -self.recovery_speed  
            elif o['active'] != o['designated'] and o['ball'][0] < 0.5:  # Recovery to back
                components['recovery_reward'][rew_index] = self.recovery_speed

            reward[rew_index] += sum(components.values())

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
