import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for optimizing shooting angles and timing under pressure near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_threshold = 0.2  # defines how close to the goal the shot must occur
        self.angle_bonus = 0.1  # bonus for optimal shooting angle
        self.timing_penalty = -0.05  # penalty for mis-timed shots

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'angle_reward': [0.0] * len(reward), 'timing_penalty': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]  # only consider x, y coordinates
            goal_pos = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0]
            dist_to_goal = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))

            # Check if within the shooting threshold distance to the goal
            if dist_to_goal <= self.shot_threshold:
                if 'action' in o and o['action'] == 5:  # Assuming 5 is the index for the shoot action
                    angle = np.dot(ball_pos, goal_pos) / (np.linalg.norm(ball_pos) * np.linalg.norm(goal_pos))
                    components['angle_reward'][rew_index] = angle * self.angle_bonus
                else:
                    components['timing_penalty'][rew_index] = self.timing_penalty

                reward[rew_index] += components['angle_reward'][rew_index] + components['timing_penalty'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
