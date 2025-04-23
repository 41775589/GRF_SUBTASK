import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on midfield dynamics."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_threshold = 0.2  # threshold for midfield area
        self.midfield_reward = 0.05  # reward for midfield control
        self.transition_reward = 0.1  # reward for transitioning between defense and attack
        self._last_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._last_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_dynamics'] = {
            "last_ball_position": self._last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle['midfield_dynamics']['last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_x = o['ball'][0]
            
            # Reward for controlling the midfield area
            if abs(ball_x) <= self.midfield_threshold:
                components["midfield_control_reward"][rew_index] = self.midfield_reward

            # Reward for ball transitioning from one side to another
            if self._last_ball_position is not None:
                if (self._last_ball_position < -self.midfield_threshold and ball_x > self.midfield_threshold) or \
                   (self._last_ball_position > self.midfield_threshold and ball_x < -self.midfield_threshold):
                    components["transition_reward"][rew_index] = self.transition_reward
            
            reward[rew_index] += components["midfield_control_reward"][rew_index] + components["transition_reward"][rew_index]
            
            self._last_ball_position = ball_x  # update last ball x position for next step

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
