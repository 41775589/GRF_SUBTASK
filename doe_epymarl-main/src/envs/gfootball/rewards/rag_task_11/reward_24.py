import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ Custom wrapper to enhance offensive play through fast-paced maneuvers and precision finishing. """

    def __init__(self, env):
        super().__init__(env)
        self.precision_bonus = 0.2  # Bonus reward for precision in finishing
        self.fast_break_bonus = 0.3  # Bonus reward for fast-paced gameplay
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_bonus": [0.0] * len(reward),
                      "fast_break_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            ball_pos = obs['ball']
            goal_pos = [1, 0]  # Assuming playing left to right
            distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(ball_pos[:2]))

            # Reward fast breaks (moving quickly towards goal direction)
            if obs['ball_direction'][0] > 0 and distance_to_goal < 0.5:
                components["fast_break_bonus"][i] = self.fast_break_bonus
                reward[i] += components["fast_break_bonus"][i]

            # Reward precision when in possession near the goal area
            if obs['ball_owned_team'] == 1 and distance_to_goal < 0.2:
                components["precision_bonus"][i] = self.precision_bonus
                reward[i] += components["precision_bonus"][i]

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
