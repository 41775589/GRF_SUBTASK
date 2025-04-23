import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive play and finishing skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the number of reward zones leading up to the opponent's goal
        self.num_zones = 5
        self.zone_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_tracker = {i: False for i in range(self.num_zones)}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.zone_tracker
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zone_tracker = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Assign additional rewards based on attacking position and proximity to opponent's goal
        for i, o in enumerate(observation):
            ball_pos = o['ball'][0]  # X-axis position of the ball
            if ball_pos > 0:  # Ball must be on opponent's half
                zone_index = min(int((ball_pos + 1) / 0.4), self.num_zones - 1)  # Divide field into zones
                if not self.zone_tracker[zone_index]:  # Reward only the first time entering the zone
                    components["offensive_play_reward"][i] = self.zone_reward
                    reward[i] += components["offensive_play_reward"][i]
                    self.zone_tracker[zone_index] = True

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
                if action == 1:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
