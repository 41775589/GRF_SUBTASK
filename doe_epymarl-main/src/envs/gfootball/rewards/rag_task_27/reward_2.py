import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to introduce a reward based on defensive abilities, focusing
    specifically on interceptions and maintaining proper defensive positioning.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define defensive zone proximity checkpoints
        self.defensive_zone_thresholds = np.linspace(-1, 0, 5)  # Defensive zones from own goal to midfield
        self.interception_bonus = 0.2  # Reward added for intercepting the ball
        self.positioning_bonus = 0.1  # Reward added for staying in a strategic defensive position
        self.captured_positions = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.captured_positions = set()
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            active_player_pos = o['right_team'][o['active']]  # Get the position of the active (controlled) player

            # Check defensive interceptions
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["interception_reward"][idx] += self.interception_bonus
                reward[idx] += components["interception_reward"][idx]

            # Check defensive positioning
            if active_player_pos[0] < 0:  # Player is on own side
                for threshold in self.defensive_zone_thresholds:
                    if active_player_pos[0] < threshold and threshold not in self.captured_positions:
                        components["positioning_reward"][idx] += self.positioning_bonus
                        reward[idx] += components["positioning_reward"][idx]
                        self.captured_positions.add(threshold)

        return reward, components

    def step(self, action):
        self.step(action)

        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
