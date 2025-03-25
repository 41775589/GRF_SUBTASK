import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a complex dribbling and sprinting reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_completion_threshold = 3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Obtain the latest observation and initialize reward components
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_and_sprint_bonus": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            
            # Check for dribbling and sprinting simultaneously, punishing losses of dribbling control
            if player_obs['sticky_actions'][8] == 1 and player_obs['sticky_actions'][9] == 1:  # sprint and dribble both active
                components['dribble_and_sprint_bonus'][rew_index] += 0.05
            if player_obs['sticky_actions'][9] == 1:
                self.sticky_actions_counter[rew_index] += 1
            
            # Reward based on maintaining dribble control over a number of steps
            if self.sticky_actions_counter[rew_index] >= self.dribble_completion_threshold:
                components['dribble_and_sprint_bonus'][rew_index] += 0.5
                self.sticky_actions_counter[rew_index] = 0  # reset counter after rewarding

            # Calculate final reward for each agent
            reward[rew_index] += components['dribble_and_sprint_bonus'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
