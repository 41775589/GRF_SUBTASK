import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding defensive actions, specifically tackles, without fouling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.tackle_reward = 0.2
        self.non_foul_bonus = 0.1
        self.yellow_card_penalty = -0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "non_foul_bonus": [0.0] * len(reward),
                      "yellow_card_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Rewards for successful tackle without a foul
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1 and o['game_mode'] in [3, 5, 6]:
                components["tackle_reward"][rew_index] += self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
                if 'yellow_card' not in o or not o['yellow_card']:  # No yellow card after tackle
                    components["non_foul_bonus"][rew_index] += self.non_foul_bonus
                    reward[rew_index] += components["non_foul_bonus"][rew_index]
            # Penalties for causing fouls
            if 'yellow_card' in o and o['yellow_card']:
                components["yellow_card_penalty"][rew_index] += self.yellow_card_penalty
                reward[rew_index] += components["yellow_card_penalty"][rew_index]

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
