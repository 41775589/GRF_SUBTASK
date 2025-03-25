import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for successful defensive actions such as sliding tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the reward for a well-timed and successful sliding tackle
        self.tackle_reward = 0.3
        # Define the extra pressure factor reward when in critical defensive zones
        self.pressure_factor_reward = 0.2

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            if obs['sticky_actions'][9] == 1:  # Checking if 'action_dribble' is active
                if obs['game_mode'] == 0:  # Normal game mode
                    if obs['ball_owned_team'] == 0:  # Left team, example considering agent in left team
                        opp_goal_dist = 1 + obs['ball'][0]  # Distance to own goal from -1 to 1
                        if opp_goal_dist < 0.2:  # If the event happens in critical defensive zone
                            components["tackle_reward"][i] = self.tackle_reward + self.pressure_factor_reward
                        else:
                            components["tackle_reward"][i] = self.tackle_reward
                        reward[i] += components["tackle_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
