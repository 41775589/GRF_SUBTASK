import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on shot precision near the opponent's goal."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Number of zones close to opponent's goal
        self._goal_zone_reward = 0.2
        self._goal_x_threshold = 0.8  # x-coordinate threshold to be near the goal
        self._goal_y_threshold = 0.2  # y-coordinate threshold for height near goal
        self._rewards_collected = np.zeros((2, self._num_zones), dtype=bool)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._rewards_collected.fill(False)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._rewards_collected.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._rewards_collected = np.array(from_pickle['CheckpointRewardWrapper'], dtype=bool)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_zone_reward": [0.0, 0.0]
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            if 'ball' in o and 'ball_owned_team' in o:
                ball_x, ball_y, _ = o['ball']
                ball_owned_team = o['ball_owned_team']

                # Reward if own team controls the ball near the opponent's goal
                if ball_owned_team == (0 if i == 0 else 1):
                    zone_index = int((ball_y + self._goal_y_threshold) / (2 * self._goal_y_threshold / self._num_zones))
                    if abs(ball_x) > self._goal_x_threshold and not self._rewards_collected[i][zone_index]:
                        # Reward only once per zone per episode
                        components['precision_zone_reward'][i] += self._goal_zone_reward
                        reward[i] += self._goal_zone_reward
                        self._rewards_collected[i][zone_index] = True

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
