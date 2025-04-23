import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward focusing on defending strategies such as tackling, 
    efficient stopping, and accurate passing under pressure.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.1
        self.stop_skill_reward = 0.05
        self.pressured_pass_reward = 0.15

    def reset(self):
        """Reset the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State saving support."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State loading support."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Adjust all rewards given by the base environment by adding defending related bonuses.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "stop_skill_reward": [0.0] * len(reward),
            "pressured_pass_reward": [0.0] * len(reward)
        }

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]

            # Assumption: Tackles are measured by high ⌈ball_owned_team⌉ changes
            if 'ball_owned_team' in o:
                if o['ball_owned_team'] == 0 and o['ball'] is not None:
                    if o['ball'][0] > 0.0:  # Assuming ball on our half increases challenge
                        components["tackle_reward"][rew_index] = self.tackle_reward

            # Checking for stopping skills by analyzing player speed
            if 'left_team_direction' in o:
                if np.any(np.linalg.norm(o['left_team_direction'], axis=1) < 0.01):  # very low movement means good stopping
                    components["stop_skill_reward"][rew_index] = self.stop_skill_reward

            # Enhanced reward if pass completed under pressure
            if 'game_mode' in o and o['game_mode'] == 2:  # Assuming code 2 relates to a high pressure situation
                components["pressured_pass_reward"][rew_index] = self.pressured_pass_reward
            
            # Sum up the rewards
            reward[rew_index] += (
                components["tackle_reward"][rew_index] +
                components["stop_skill_reward"][rew_index] +
                components["pressured_pass_reward"][rew_index]
            )

        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, applying the adjusted rewards and collecting metrics.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
