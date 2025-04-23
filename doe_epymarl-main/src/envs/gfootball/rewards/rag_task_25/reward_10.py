import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling and sprinting performance."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_skill_threshold = 0.5  # placeholder value
        self.dribbling_skill_increment = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["dribbling_reward"][rew_index] = 0.0

            # Reward for dribbling and sprinting
            dribbling_action_active = o['sticky_actions'][9]  # Index 9 is for dribbling
            sprint_action_active = o['sticky_actions'][8]  # Index 8 is for sprinting

            if dribbling_action_active and sprint_action_active and o['ball_owned_team'] == 1:
                components["dribbling_reward"][rew_index] += self.dribbling_skill_increment

            # Update the reward for the current player based on dribbling
            reward[rew_index] += components["dribbling_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Store sticky actions usage for analysis
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active
        return observation, reward, done, info
