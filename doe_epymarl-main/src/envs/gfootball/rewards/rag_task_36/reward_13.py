import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for dribbling maneuvers combined with dynamic transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Reward for starting and stopping dribbles effectively
            if o['sticky_actions'][9] == 1:  # action_dribble
                components["dribbling_reward"][i] += 0.05
            else:
                components["dribbling_reward"][i] -= 0.025

            # Reward for maintaining an appropriate position transitioning between defense and offense
            if 'left_team' in o:
                player_pos = o['left_team'][o['active']]
            else:
                player_pos = o['right_team'][o['active']]

            # Ideally, midfielders (considered here as transitional players) should be around the center
            if 0.25 < player_pos[0] < 0.75:
                components["positioning_reward"][i] += 0.1
            else:
                components["positioning_reward"][i] -= 0.05

            reward[i] += components["dribbling_reward"][i] + components["positioning_reward"][i]

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
