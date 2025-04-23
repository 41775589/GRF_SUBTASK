import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper adding rewards based on effective mid to long-range passing, promoting strategic ball distribution and control."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passing_distance_threshold = 0.3  # Threshold for long passes
        self._passing_effectiveness_reward = 0.05
        self._high_pass_action_index = 9  # Assuming index 9 corresponds to 'action_high_pass'
        self._long_pass_action_index = 8  # Assuming index 8 corresponds to 'action_long_pass'
        self._last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self._last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle.get('last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_effectiveness_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            current_ball_pos = np.array(o['ball'][:2])
            current_ball_distance = np.linalg.norm(current_ball_pos)

            if self._last_ball_position is not None:
                distance_covered = np.linalg.norm(current_ball_pos - self._last_ball_position)
            else:
                distance_covered = 0
            
            is_long_pass = o['sticky_actions'][self._long_pass_action_index]
            is_high_pass = o['sticky_actions'][self._high_pass_action_index]

            # Check for a successful long/high pass
            if distance_covered > self._passing_distance_threshold and (is_long_pass or is_high_pass):
                components["passing_effectiveness_reward"][rew_index] = self._passing_effectiveness_reward
                reward[rew_index] += components["passing_effectiveness_reward"][rew_index]

            # Update last ball position
            self._last_ball_position = current_ball_pos

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        for i in range(len(self.sticky_actions_counter)):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
