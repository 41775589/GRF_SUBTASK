import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on incentivizing long passes with precision and varying conditions."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_worth = 0.1
        self.pass_distance_threshold = 0.2  # Threshold for considering a action as a long pass
        self.pass_accuracy_threshold = 0.05  # Threshold for considering a pass accurate
        self.long_pass_count = np.zeros(2)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the reward environment and counters."""
        self.long_pass_count.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the wrapper state."""
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['long_pass_count'] = self.long_pass_count
        return to_pickle

    def set_state(self, from_pickle):
        """Deserialize the wrapper state."""
        from_pickle = self.env.set_state(from_pickle)
        self.long_pass_count = from_pickle['long_pass_count']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards based on long and precise pass criteria."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o:
                ball_direction_norm = np.linalg.norm(o['ball_direction'][:2])

                # Check for a long and accurate pass
                if ball_direction_norm > self.pass_distance_threshold:
                    ball_distance_error = np.linalg.norm(o['ball'] - o['left_team'][o['ball_owned_player']][:2])
                    if ball_distance_error < self.pass_accuracy_threshold:
                        reward[rew_index] += self.pass_worth
                        components["long_pass_reward"][rew_index] = self.pass_worth
                        self.long_pass_count[rew_index] += 1

        return reward, components

    def step(self, action):
        """Acts in environment with specified action, handles rewards and component details."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Add sticky actions info, if available
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
