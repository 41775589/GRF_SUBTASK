import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for optimal shooting techniques near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining checkpoints to assess angle optimization and shooting distance
        self.angle_coefficient = 0.1  # Coefficient that balances the importance of shooting at correct angles
        self.distance_coefficient = 0.05  # Coefficient for distance to goal
        self.goal_distance_threshold = 0.2  # Min distance to goal to consider shot valid

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
            "shooting_angle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] == obs['active']:
                # Calculate distance to goal
                goal_position = [1, 0]  # Assuming right goal is at (1,0)
                ball_position = obs['ball'][:2]
                distance_to_goal = np.linalg.norm(np.array(ball_position) - np.array(goal_position))

                if distance_to_goal < self.goal_distance_threshold:
                    # Assuming optimal shooting direction is toward the goal center
                    ball_direction = obs['ball_direction'][:2]
                    optimal_direction = np.array(goal_position) - np.array(ball_position)
                    angle_diff = np.dot(ball_direction, optimal_direction) / (np.linalg.norm(ball_direction) * np.linalg.norm(optimal_direction))
                    angle_diff = np.clip(angle_diff, -1, 1)  # Clip to handle numerical errors
                    angle_reward = np.arccos(angle_diff) / np.pi  # Normalize angle diff to get a reward between 0 and 1

                    # Compute distance and angle rewards
                    components['shooting_angle_reward'][idx] = self.angle_coefficient * (1 - angle_reward) + self.distance_coefficient / (distance_to_goal + 0.1)

        for idx in range(len(reward)):
            reward[idx] += components['shooting_angle_reward'][idx]

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
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
