import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward signal for defensive actions in football.
    This includes intercepting the ball, preventing goals, and maintaining formation
    under high-pressure defensive scenarios.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.2
        self.prevent_goal_reward = 0.3
        self.formation_maintenance_reward = 0.1

    def reset(self):
        """
        Reset the reward wrapper state for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of this reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of this reward wrapper.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Enhance the reward based on defensive actions taken by the agent.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "prevent_goal_reward": [0.0] * len(reward),
            "formation_maintenance_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for interception
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] != o['active']:
                components["interception_reward"][rew_index] = self.interception_reward
                reward[rew_index] += components["interception_reward"][rew_index]

            # Check for preventing goal (assuming agent is close to their goal area)
            if o['left_team'][o['active']][0] < -0.5 and abs(o['ball'][0] - o['left_team'][o['active']][0]) < 0.1:
                components["prevent_goal_reward"][rew_index] = self.prevent_goal_reward
                reward[rew_index] += components["prevent_goal_reward"][rew_index]

            # Maintain formation (simple distance based reward for staying close to designated position)
            designated = o['designated']
            dist_to_designated = np.linalg.norm(np.array(o['left_team'][o['active']]) - np.array(o['left_team'][designated]))
            if dist_to_designated < 0.1:
                components["formation_maintenance_reward"][rew_index] = self.formation_maintenance_reward
                reward[rew_index] += components["formation_maintenance_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step in the environment and process reward adjustments.
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
