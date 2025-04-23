import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for executing long and accurate passes."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Dividing the field into 5 zones for simplicity
        self.long_pass_threshold = 0.3  # Minimum distance to count as a long pass (normalized coords.)
        self.long_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = np.zeros((2, self._num_zones), dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = np.zeros((2, self._num_zones), dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_passes'] = self.passes_completed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passes_completed = from_pickle['CheckpointRewardWrapper_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for long passes
            if o['ball_owned_team'] == 0 or o['ball_owned_team'] == 1:  # Check if either team has the ball
                ball_pos_prev = np.array(o['ball'])
                ball_pos_next = ball_pos_prev + np.array(o['ball_direction'])

                if np.linalg.norm(ball_pos_next - ball_pos_prev) > self.long_pass_threshold:
                    # Check which zone the long pass was to
                    team_index = o['ball_owned_team']
                    zone_dist = np.linspace(-1, 1, self._num_zones + 1)
                    start_zone = np.digitize([ball_pos_prev[0]], zone_dist).item() - 1
                    end_zone = np.digitize([ball_pos_next[0]], zone_dist).item() - 1

                    if start_zone != end_zone:  # Must be a pass to a different zone to be rewarded
                        self.passes_completed[team_index][end_zone] += 1
                        components['long_pass_reward'][rew_index] = self.long_pass_reward
                        reward[rew_index] += components['long_pass_reward'][rew_index]

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
