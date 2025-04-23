import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards developing offensive strategies including shooting, dribbling,
    and passing to break through the opponent's defense.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.2
        self.passing_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, r in enumerate(reward):
            o = observation[idx]
            game_mode = o['game_mode']

            # Adding reward for successful shooting towards goal
            if game_mode == 6 and o['ball_owned_team'] == 1:
                # Assuming ball_owned_team of 1 is the offensive team
                components['shooting_reward'][idx] += self.shooting_reward
                reward[idx] += components['shooting_reward'][idx]

            # Adding reward for dribbling (assuming action_dribble sticky action presence)
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 1:
                components['dribbling_reward'][idx] += self.dribbling_reward
                reward[idx] += components['dribbling_reward'][idx]

            # Adding reward for passing long or high
            if (o['sticky_actions'][4] == 1 or o['sticky_actions'][8] == 1) and o['ball_owned_team'] == 1:
                # Assuming action_right and action_sprint might indicate a long or high pass
                components['passing_reward'][idx] += self.passing_reward
                reward[idx] += components['passing_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Check and report on sticky actions continuity
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
