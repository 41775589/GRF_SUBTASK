import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on optimizing shooting angles and timing
    in high-pressure scenarios near the goal.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_threshold = 0.1  # Close proximity to the goal to consider high pressure
        self.shooting_angle_coefficient = 0.5  # Coefficient for rewarding good shooting angles

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
        components = {"base_score_reward": reward.copy(),
                      "shooting_angle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            # Determine proximity to the goal (considered high-pressure if close to opponent's goal)
            if ball_pos[0] > (1 - self.distance_threshold):
                # Calculate the angle of approach to the goal
                goal_center_y = 0
                y_dist_to_goal = goal_center_y - ball_pos[1]
                angle = np.arctan2(y_dist_to_goal, 1 - ball_pos[0])
                angle_degree = abs(np.degrees(angle))

                # Optimal shooting angle is between 20 to 45 degrees
                if 20 <= angle_degree <= 45:
                    angle_reward = self.shooting_angle_coefficient * (45 - abs(angle_degree - 20))
                    components["shooting_angle_reward"][rew_index] = angle_reward
                    reward[rew_index] += angle_reward

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
