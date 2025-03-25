import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a structured reward based on positions,
    ball possessions, and strategic actions to foster offensive play and team coordination.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._target_position_reward = 0.05
        self._pass_completion_reward = 0.1
        self._goal_scoring_opportunity_reward = 0.15

    def reset(self):
        """
        Resets environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Reward function encouraging offensive strategies, team coordination, possession, and strategic positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward),
                      "goal_opportunity_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, current_reward in enumerate(reward):
            o = observation[rew_index]
            ball_position = o['ball'][:2]  # x, y coordinates

            # Reward for moving closer to opponent goal
            if o['ball_owned_team'] == 1:  # if possessed by right team
                distance_to_goal = (1 - o['ball'][0])
                components["positioning_reward"][rew_index] = self._target_position_reward * max(0, distance_to_goal)

            # Reward for successful passes
            if o['ball_owned_team'] == 1 and o['sticky_actions'][9]:  # dribbling action is used as a proxy for passing
                components["passing_reward"][rew_index] = self._pass_completion_reward

            # Reward for creating goal scoring opportunities
            if o['game_mode'] in [3, 4]:  # free kicks or corners, implying an offense near the box
                components["goal_opportunity_reward"][rew_index] = self._goal_scoring_opportunity_reward

            # Update the reward
            reward[rew_index] += sum(components[k][rew_index] for k in components)

        return reward, components

    def step(self, action):
        """
        Takes a step using action, calculates and returns the observation, reward, done, and info.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        info.update({f'component_{k}': sum(v) for k, v in components.items()})
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            if agent_obs:
                for i, act in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += int(act)
        return observation, reward, done, info
