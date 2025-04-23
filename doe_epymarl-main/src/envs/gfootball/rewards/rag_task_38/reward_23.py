import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on initiating counterattacks with long passes and quick transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward parameters
        self.long_pass_reward = 0.2
        self.quick_transition_reward = 0.3
        self.last_ball_position = np.zeros(3)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper_last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0],
                      "quick_transition_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o.get('ball')
            
            # Calculate distance covered by the ball in one step
            distance_moved = np.linalg.norm(ball_position - self.last_ball_position)
            self.last_ball_position = ball_position.copy()

            # Check for a long pass
            if distance_moved > 0.3:  # Threshold to determine a "long" pass
                components["long_pass_reward"][rew_index] = self.long_pass_reward
                reward[rew_index] += components["long_pass_reward"][rew_index]

            # Transition Logic: reward for quickly moving from own half to opposite half while having the ball
            if o.get('ball_owned_team') == o.get('active') and ball_position[0] > 0 and self.last_ball_position[0] <= 0:
                components["quick_transition_reward"][rew_index] = self.quick_transition_reward
                reward[rew_index] += components["quick_transition_reward"][rew_index]

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
        return observation, reward, done, info
