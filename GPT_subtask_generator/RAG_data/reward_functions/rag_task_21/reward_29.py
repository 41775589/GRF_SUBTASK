import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on defensive skills and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to adjust the significance of the defensive events
        self.interception_reward = 0.3
        self.positioning_reward = 0.1
        self.defensive_region_threshold = 0.5  # delineates the defensive area of the field

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for checkpointing between episodes."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state for checkpointing between episodes."""
        from_pickle = self.env.set_state(state)
        # No specific state needed in this wrapper for setting up
        return from_pickle

    def reward(self, reward):
        """Calculate and return augmented reward based on defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage interceptions: Check if this agent intercepted the ball
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                'designated' in o and o['active'] == o['designated'] and
                'prev_ball_owned_team' in o and o['prev_ball_owned_team'] == 1):
                reward[rew_index] += self.interception_reward
                components["interception_reward"][rew_index] = self.interception_reward

            # Encourage good defensive positioning in own half
            player_pos = o['left_team'][o['active']]
            if player_pos[0] < -self.defensive_region_threshold:
                reward[rew_index] += self.positioning_reward
                components["positioning_reward"][rew_index] = self.positioning_reward

        return reward, components

    def step(self, action):
        """Take a step in the environment and augment reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        return observation, reward, done, info
