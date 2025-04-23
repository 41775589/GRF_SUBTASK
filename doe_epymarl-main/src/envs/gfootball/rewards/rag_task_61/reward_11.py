import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic positioning rewards based on precise timing and positioning for possession changes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._possession_changes = {}
        self._previous_ball_owner = None  # Track which team last owned the ball
        self._timing_reward = 0.2  # Reward for effective timing of actions
        self._positioning_reward = 0.1  # Reward for strategic positioning

    def reset(self):
        self._possession_changes = {}
        self._previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._possession_changes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._possession_changes = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "timing_reward": [0.0] * len(reward), 
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            current_ball_owner = o['ball_owned_team']
            
            # Check for possession change
            if self._previous_ball_owner is not None and self._previous_ball_owner != current_ball_owner:
                reward[rew_index] += self._timing_reward
                components["timing_reward"][rew_index] += self._timing_reward
            
            # Calculate positioning rewards based on strategic game play
            if current_ball_owner == 0:  # Assuming this agent's team is '0'
                dist_to_goal = np.sqrt((o['ball'][0] - 1) ** 2 + (o['ball'][1]) ** 2)  # Distance to opponent's goal
                reward[rew_index] += (0.1 - dist_to_goal) * self._positioning_reward  # Better position, higher reward
                components["positioning_reward"][rew_index] += (0.1 - dist_to_goal) * self._positioning_reward

            self._previous_ball_owner = current_ball_owner  # Update for next step check

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
