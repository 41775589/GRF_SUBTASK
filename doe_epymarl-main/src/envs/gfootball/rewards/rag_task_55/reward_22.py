import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for defensive maneuvers, focusing on standing and sliding tackles without fouling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_successful = 0
        self.no_foul = 0
        self.standing_tackle_reward = 0.5
        self.sliding_tackle_reward = 0.7
        self.no_foul_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_successful = 0
        self.no_foul = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['Tackles'] = {
            'tackle_successful': self.tackle_successful,
            'no_foul': self.no_foul
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_successful = from_pickle['Tackles']['tackle_successful']
        self.no_foul = from_pickle['Tackles']['no_foul']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "standing_tackle_reward": [0.0] * len(reward),
                      "sliding_tackle_reward": [0.0] * len(reward),
                      "no_foul_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful standing tackle without committing a foul
            if o['sticky_actions'][7] == 1 and self.tackle_successful == 0:  # 7 is the index for a standing tackle action
                self.tackle_successful += 1
                reward[rew_index] += self.standing_tackle_reward
                components["standing_tackle_reward"][rew_index] = self.standing_tackle_reward
            
            if o['sticky_actions'][8] == 1 and self.tackle_successful == 0:  # 8 is the index for a sliding tackle action
                self.tackle_successful += 1
                reward[rew_index] += self.sliding_tackle_reward
                components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward

            # Additional reward for not committing a foul
            if self.tackle_successful > 0 and self.no_foul == 0:
                self.no_foul += 1
                reward[rew_index] += self.no_foul_reward
                components["no_foul_reward"][rew_index] = self.no_foul_reward

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
