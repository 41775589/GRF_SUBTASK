import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on learning energy conservation techniques through proficient usage of Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.previous_actions = np.zeros((2, 10), dtype=int)  # Assumes two agents
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_actions = np.zeros((2, 10), dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.previous_actions.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_actions = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        # Access last observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "energy_conservation_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage stop if the sprint action was taken before, and then stopped
            sprint_index = 8
            if self.previous_actions[rew_index][sprint_index] == 1 and o['sticky_actions'][sprint_index] == 0:
                components["energy_conservation_reward"][rew_index] += 0.1  # Reward for stopping sprint
                reward[rew_index] += components["energy_conservation_reward"][rew_index]

            # Likewise, encourage stopping any movement if previously moving
            for action_index, was_active in enumerate(self.previous_actions[rew_index]):
                if was_active == 1 and o['sticky_actions'][action_index] == 0 and action_index != sprint_index:
                    components["energy_conservation_reward"][rew_index] += 0.05  # Reward for stopping movement
                    reward[rew_index] += components["energy_conservation_reward"][rew_index]

            # Update previous actions to current
            self.previous_actions[rew_index] = o['sticky_actions'].copy()

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
