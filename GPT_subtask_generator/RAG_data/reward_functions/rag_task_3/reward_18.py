import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on practicing effective shooting."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_threshold_distance = 0.3  # distance within which shots are highly rewarded
        self.shot_power_reward = 0.5  # additional reward for powerful shots
        self.pressure_factor = 0.2  # reward modification based on defensive pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_practice_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]  # assuming single agent
        ball_pos = np.array(o['ball'][:2])
        goal_pos = np.array([1, 0])  # approximate goal center position on right side

        # Calculate Euclidean distance to the opponent's goal
        distance_to_goal = np.linalg.norm(ball_pos - goal_pos)

        # Shoot if close to the goal and ball is owned
        if distance_to_goal <= self.shot_threshold_distance and o['ball_owned_team'] == 0:
            components["shooting_practice_reward"][0] = self.shot_power_reward
            # Reward for shooting under pressure
            num_opponents_nearby = np.sum(np.linalg.norm(o['left_team'] - ball_pos, axis=1) < 0.15)
            components["shooting_practice_reward"][0] += num_opponents_nearby * self.pressure_factor
        
        reward[0] += sum(components["shooting_practice_reward"])

        return reward, components

    def step(self, action):
        # This default implementation collects observation, rewards, and done flag from the environment
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
