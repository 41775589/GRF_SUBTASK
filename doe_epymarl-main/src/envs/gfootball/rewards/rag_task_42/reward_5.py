import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on midfield dynamics and transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters defining the midfield range and dynamics
        self.midfield_range = [-0.2, 0.2]  # Represents midfield section of the pitch
        self.reward_for_midfield_play = 0.05
        self.last_ball_position = None
        self.midfield_transitions = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.midfield_transitions = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_dynamics": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Calculate midfield dynamics reward and transitions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x = o['ball'][0]
            midfield_low, midfield_high = self.midfield_range

            # Check if the play is in the midfield
            if midfield_low <= ball_x <= midfield_high:
                # Increase reward for playing in the midfield
                components["midfield_dynamics"][rew_index] = self.reward_for_midfield_play
                reward[rew_index] += components["midfield_dynamics"][rew_index]

                # Track transitions into and out of the midfield
                if self.last_ball_position is not None:
                    if (self.last_ball_position < midfield_low and ball_x >= midfield_low) or \
                       (self.last_ball_position > midfield_high and ball_x <= midfield_high):
                        self.midfield_transitions += 1

            self.last_ball_position = ball_x
        
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
                self.sticky_actions_counter[i] += action
        info["transitions_in_midfield"] = self.midfield_transitions
        return observation, reward, done, info
