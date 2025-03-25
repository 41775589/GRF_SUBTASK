import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards for mastering long passes in football game."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_thresholds = np.linspace(0.4, 0.9, 5)  # Define thresholds for a successful long pass
        self.pass_completion_reward = 0.5
        self.long_pass_attempt_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Calculates reward depth on long pass completion and attempts under differing match conditions.
        
        Each attempt to make a long pass earns a small positive reward. Completing a pass longer than
        predefined thresholds gives a higher reward. This pattern trains precise and situationally
        adapted long passes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_attempt_reward": [0.0] * len(reward),
                      "long_pass_completion_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            ball_pos = o['ball']
            ball_owner = o['ball_owned_player']

            # Check if the agent is actively doing a long pass
            ball_travel_distance = np.linalg.norm(ball_pos[-3:-1]) - np.linalg.norm(ball_pos[:2])
            is_long_pass = ball_travel_distance > 0.2  # Arbitrary threshold for long pass attempts

            if is_long_pass:
                components['long_pass_attempt_reward'][idx] = self.long_pass_attempt_reward
                reward[idx] += self.long_pass_attempt_reward

                # Check if the pass crosses one of the preset thresholds for a bonus reward
                for threshold in self.pass_thresholds:
                    if ball_travel_distance > threshold and o['ball_owned_player'] == ball_owner:
                        components['long_pass_completion_reward'][idx] += self.pass_completion_reward
                        reward[idx] += self.pass_completion_reward

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
