import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling and dodging the goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_bonus = 0.05

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
        # Grab the latest observation
        observation = self.env.unwrapped.observation()
        assert len(reward) == len(observation), "Reward and observation must be of same length"
        
        components = {"base_score_reward": reward.copy(), "dribble_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['active'] == -1:  # In case no active player
                continue

            # Check if the player is near the opposing goalkeeper, and executing dribble or direction change
            goalkeeper_position = o['right_team'][0]  # Assuming goalkeeper is the first in right team list
            player_position = o['left_team'][o['active']]
            distance_to_gk = np.linalg.norm(player_position[:2] - goalkeeper_position[:2])

            # Check if dribbling and close to goalkeeper
            is_dribbling = o['sticky_actions'][9] == 1
            if distance_to_gk < 0.1 and is_dribbling:
                components["dribble_reward"][rew_index] += self.dribble_bonus
                reward[rew_index] += components["dribble_reward"][rew_index]

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
