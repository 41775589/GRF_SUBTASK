import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specializes in optimizing shooting angles and timing under high-pressure scenarios near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_zone_threshold = 0.2  # Threshold to consider 'near goal'
        self.base_reward_scaling = 1.0
        self.shooting_angle_reward = 2.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_angle_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            components["base_score_reward"][i] *= self.base_reward_scaling

            if o['game_mode'] in [0, 6]:  # Normal play or Penalty
                ball_x = o['ball'][0]
                ball_owned_team = o['ball_owned_team']
                if ball_owned_team == o['active']:
                    # Reward being close to the opponent's goal and having the ball
                    if abs(ball_x) > (1 - self.goal_zone_threshold) and o['active'] == o['ball_owned_player']:
                        angle_to_goal = np.arctan2(abs(o['ball'][1]), 1 - abs(ball_x))
                        components["shooting_angle_reward"][i] = self.shooting_angle_reward * (np.pi/2 - angle_to_goal)

                reward[i] += components["shooting_angle_reward"][i]

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
