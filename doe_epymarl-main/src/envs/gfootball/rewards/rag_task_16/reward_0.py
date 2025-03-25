import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on high pass execution quality in a football game scenario."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = self.env.get_state(to_pickle)
        state_info['sticky_actions_counter'] = self.sticky_actions_counter
        return state_info
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # This is the base reward given by the environment for the current frame.
        components = {"base_score_reward": reward.copy()}
        
        # Retrieve the current observation from the wrapped environment.
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation), "Mismatched reward and observation lengths."

        for idx, obs in enumerate(observation):
            # Analyzing if high pass was executed
            if obs['ball_owned_team'] == 1 and obs['ball'][2] > 0.1:  # Assuming a high pass criteria: ball z-coord > 0.1
                # We will add a reward based on the verticality and speed of the ball with respect to the opponent's goal.
                height_bonus = obs['ball'][2] ** 2  # Rewarding square of the height for loftedness
                trajectory_bonus = (1 - np.abs(obs['ball_direction'][1])) * 0.5  # Reward for forwarding direction
                power_assessment = np.abs(obs['ball_direction'][2])  # Reward for speed in the pass

                # Total additional reward for high passes
                additional_reward = (height_bonus + trajectory_bonus + power_assessment) * 0.1
                reward[idx] += additional_reward
                components.setdefault("high_pass_quality_reward", []).append(additional_reward)
            else:
                components.setdefault("high_pass_quality_reward", []).append(0.0)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
