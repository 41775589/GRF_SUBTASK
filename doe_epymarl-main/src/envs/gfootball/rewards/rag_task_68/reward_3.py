import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward, focused on offensive football skills including shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Reward settings for specific actions
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.3
        self.passing_reward = 0.5

    def reset(self):
        """Reset the environment and sticky actions count at the start of a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the environment state including the sticky actions counter."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state and restore the sticky actions counter."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Calculate the modified rewards based on specific offensive actions taken by the agents."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Check if the controlled team owns the ball
                if 'game_mode' in o and o['game_mode'] == 6:  # mode 6 is Penalty
                    components["shooting_reward"][i] = self.shooting_reward
                if 'sticky_actions' in o:
                    if o['sticky_actions'][9] == 1:  # index 9 is dribbling
                        components["dribbling_reward"][i] = self.dribbling_reward
                    if o['sticky_actions'][0] == 1 or o['sticky_actions'][4] == 1:  # indices 0 and 4 are long lateral movements, simulating passing
                        components["passing_reward"][i] = self.passing_reward

            # Aggregate all component rewards to form the final reward for the agent
            reward[i] += (components["shooting_reward"][i] +
                           components["dribbling_reward"][i] +
                           components["passing_reward"][i])

        return reward, components

    def step(self, action):
        """Process the environment step and compute the reward accordingly."""
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
