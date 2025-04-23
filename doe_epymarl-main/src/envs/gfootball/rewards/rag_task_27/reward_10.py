import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive play and interceptions in football games."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking which sticky actions are being used
        self.interception_count = [0] * 3  # Number of interceptions for each agent
        self.defensive_positioning_scores = [0.0] * 3  # Positioning scoring for defense
        self.interception_reward = 0.5  # reward for successful interception
        self.positioning_reward = 0.1  # reward increment for good defensive positioning

    def reset(self):
        """
        reset the environment and the reward-related parameters.
        """
        # Reset interception and positioning trackers
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_count = [0] * 3
        self.defensive_positioning_scores = [0.0] * 3
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get state information from the wrapped environment.
        """
        to_pickle['interception_count'] = self.interception_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set state information in the wrapped environment.
        """
        from_pickle = self.env.set_state(state)
        self.interception_count = from_pickle['interception_count']
        return from_pickle

    def reward(self, reward):
        """
        Reward function that enhances defensive capabilities by rewarding interceptions and positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "interception_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Reward for intercepting the ball from an opponent
            if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] == obs['active']:
                if self.interception_count[rew_index] < 3:  # Max 3 interceptions rewarded per agent per episode
                    components["interception_reward"][rew_index] = self.interception_reward
                    self.interception_count[rew_index] += 1
                    reward[rew_index] += components["interception_reward"][rew_index]

            # Reward for good defensive positioning (proximity to opposing player who has the ball)
            if obs['ball_owned_team'] == 1 and 'right_team' in obs:
                min_distance = min(np.linalg.norm(
                    obs['left_team'][obs['active']] -
                    opp_position) for opp_position in obs['right_team'])
                # Defensive reward based on proximity - closer to the ball-holder, higher the reward
                if min_distance < 0.2:
                    components["defensive_positioning_reward"][rew_index] = self.positioning_reward / (min_distance + 0.1)
                    reward[rew_index] += components["defensive_positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Takes an action in the environment and applies the reward wrapper logic.
        """
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
