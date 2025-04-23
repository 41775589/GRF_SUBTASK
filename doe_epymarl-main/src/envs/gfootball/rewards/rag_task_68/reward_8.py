import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances offensive skills: shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_bonus = 0.1
        self.dribbling_bonus = 0.05
        self.passing_bonus = 0.07
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_bonus": [0.0] * len(reward),
                      "dribbling_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check if there is a shooting opportunity
            if o['game_mode'] == 6:  # Penalty Game Mode for simplicity as a proxy for shooting
                components['shooting_bonus'][rew_index] = self.shooting_bonus
                reward[rew_index] += components['shooting_bonus'][rew_index]

            # Dribbling skill improvement
            if o['sticky_actions'][9] == 1:  # Dribbling action is active
                components['dribbling_bonus'][rew_index] = self.dribbling_bonus
                reward[rew_index] += components['dribbling_bonus'][rew_index]

            # Passing skill incentive, monitoring long (index 9) or high passes (index 8)
            if o['sticky_actions'][8] == 1: # Assuming high passes are index 8
                components['passing_bonus'][rew_index] = self.passing_bonus
                reward[rew_index] += components['passing_bonus'][rew_index]
            elif o['sticky_actions'][6] == 1:  # Assuming index 6 for long passes, if applicable
                components['passing_bonus'][rew_index] = self.passing_bonus * 1.5  # Slightly higher reward for long passes
                reward[rew_index] += components['passing_bonus'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = (self.sticky_actions_counter[i] * 0.9 + action)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info

# Explanation of the enhanced reward components:
# Component explanations:
# shooting_bonus - encourages taking shooting opportunities, especially in game modes like penalties
# dribbling_bonus - rewards continuous dribbling, promoting evasive maneuvers around defenders
# passing_bonus - incentivizes executing tactical passes, particularly long and high passes that can break through defensive lines
