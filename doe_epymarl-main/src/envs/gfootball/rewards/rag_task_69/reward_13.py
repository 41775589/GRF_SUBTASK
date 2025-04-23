import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to add detailed rewards for offensive strategies, including accurate shooting,
    effective dribbling, and smart passing.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_positions = [0.7, 0.9]  # ideal shooting ranges
        self.passing_reward = 0.3  # reward for effective passing
        self.dribbling_reward = 0.1  # reward for effective dribbling

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
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            x_ball, y_ball, _ = o['ball']
            active_player = o['active']
            has_ball = o['ball_owned_team'] == 0

            # Reward for getting the ball into good shooting positions
            if has_ball and x_ball >= self.shooting_positions[0] and x_ball <= self.shooting_positions[1]:
                if reward[rew_index] > 0:  # assuming reward > 0 correlates to a goal or good attempt
                    components['shooting_reward'][rew_index] = self.passing_reward

            # Reward for successful passes, particularly through long or high passes
            if o['game_mode'] in [5, 6]:  # Assuming these modes are indicative of post-pass plays
                components['passing_reward'][rew_index] = self.passing_reward

            # Reward for dribbling: counting dribble actions in sticky actions
            if o['sticky_actions'][9]:  # index 9 corresponds to 'action_dribble'
                components['dribbling_reward'][rew_index] = self.dribbling_reward

            # Aggregate all components for the final reward
            reward[rew_index] += (components['shooting_reward'][rew_index] +
                                  components['passing_reward'][rew_index] +
                                  components['dribbling_reward'][rew_index])

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
