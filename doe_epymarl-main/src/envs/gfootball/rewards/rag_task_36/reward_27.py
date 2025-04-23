import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds an adaptive dribble and positioning reward based on dynamic transitions between defense and offense."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positioning_reward = 0.05
        self.dribbling_reward = 0.1

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
        components = {'base_score_reward': reward.copy(), 'positioning_reward': [0.0] * len(reward), 'dribbling_reward': [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for effective dribbling
            if o['sticky_actions'][9] == 1:  # Check if dribbling action is active
                components['dribbling_reward'][rew_index] = self.dribbling_reward
                reward[rew_index] += components['dribbling_reward'][rew_index]

            # Reward for dynamic positioning: motivate movements towards ball when defending and towards goal when attacking
            ball_position = o['ball']
            player_position = o['right_team'][rew_index] if o['ball_owned_team'] == 0 else o['left_team'][rew_index]
            goal_position = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0]

            # Distance to ball and goal to determine dynamic positioning
            distance_to_ball = np.linalg.norm(np.array(ball_position[:2]) - np.array(player_position[:2]))
            distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(player_position[:2]))
            
            # If closer to ball when defending or closer to goal when attacking, grant reward
            if o['ball_owned_team'] != -1 and (distance_to_ball < 0.3 or distance_to_goal < 0.3):
                components['positioning_reward'][rew_index] = self.positioning_reward
                reward[rew_index] += components['positioning_reward'][rew_index]

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
