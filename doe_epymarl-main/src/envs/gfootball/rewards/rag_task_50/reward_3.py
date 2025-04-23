import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for completing accurate long passes between specific field zones."""

    def __init__(self, env):
        super().__init__(env)
        # Zones are defined in terms of x-values to simplify field division.
        self.zones = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]
        self.pass_success_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            valid_pass = self.detect_valid_long_pass(obs)
            if valid_pass:
                components["long_pass_reward"][rew_index] = self.pass_success_reward
                reward[rew_index] += self.pass_success_reward

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

    def detect_valid_long_pass(self, observation):
        """Check if a long pass from one zone to another has been successfully made."""
        if observation['ball_owned_team'] == -1:
            return False
      
        ball_pos_x = observation['ball'][0]
        player_pos_x = observation['right_team' if observation['ball_owned_team'] == 1 else 'left_team'][observation['ball_owned_player']][0]

        start_zone = self.get_zone_for_position(player_pos_x)
        end_zone = self.get_zone_for_position(ball_pos_x)

        # Consider it a long pass if it crosses at least one zone boundary
        return abs(start_zone - end_zone) > 1

    def get_zone_for_position(self, pos_x):
        """Determine the zone based on the x position."""
        for i, (start, end) in enumerate(self.zones):
            if start <= pos_x < end:
                return i
        return len(self.zones) - 1  # Last zone
