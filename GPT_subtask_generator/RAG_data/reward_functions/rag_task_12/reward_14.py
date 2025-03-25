import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adjusts the reward to motivate specific behaviors for a hybrid midfielder/advance defender agent."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for high pass and long pass.
            if o['sticky_actions'][7] or o['sticky_actions'][8]:  # Assuming indices for high pass and long pass
                components.setdefault("pass_reward", [0.0] * len(reward))
                components["pass_reward"][rew_index] = 0.2
                reward[rew_index] += components["pass_reward"][rew_index]
            
            # Reward for dribbling under pressure.
            if o['sticky_actions'][9]:  # Assuming index for dribble
                components.setdefault("dribble_reward", [0.0] * len(reward))
                components["dribble_reward"][rew_index] = 0.1
                reward[rew_index] += components["dribble_reward"][rew_index]
                
            # Reward for effective sprint management.
            if o['sticky_actions'][8] and np.sum(o['sticky_actions'][0:7]) == 0:  # Assuming index for sprint
                components.setdefault("sprint_reward", [0.0] * len(reward))
                components["sprint_reward"][rew_index] = 0.05
                reward[rew_index] += components["sprint_reward"][rew_index]
            if o['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] > 3:  # stop sprint
                components.setdefault("stop_sprint_reward", [0.0] * len(reward))
                components["stop_sprint_reward"][rew_index] = 0.05
                reward[rew_index] += components["stop_sprint_reward"][rew_index]

            self.sticky_actions_counter = o['sticky_actions'].copy()

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
