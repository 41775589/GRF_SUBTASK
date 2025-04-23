import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on shooting accuracy from the central field."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define central field parameters (X coordinate ranges considered central)
        self.central_field_left = -0.25
        self.central_field_right = 0.25
        self.reward_for_precision = 0.2

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Apply additional rewards for accurate shooting from central field
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # only consider the x-coordinate

            # Check if the ball is in the central field when a goal is scored
            if ball_position >= self.central_field_left and ball_position <= self.central_field_right:
                if o['game_mode'] == 2 and o['score'][0] > 0:  # 2 denotes a goal scored in game modes
                    components["accuracy_reward"][rew_index] = self.reward_for_precision
                    reward[rew_index] += components["accuracy_reward"][rew_index]
        
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
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
