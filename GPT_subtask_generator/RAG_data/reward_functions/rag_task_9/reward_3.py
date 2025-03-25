import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for training offensive skills like passing, shooting, and dribbling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.2
        self.shot_reward = 0.3
        self.dribble_reward = 0.1
        self.sprint_reward = 0.05

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
        # Base reward from environment
        components = {"base_score_reward": reward.copy()}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        components["pass_reward"] = [0.0] * len(reward)
        components["shot_reward"] = [0.0] * len(reward)
        components["dribble_reward"] = [0.0] * len(reward)
        components["sprint_reward"] = [0.0] * len(reward)

        for rew_index, o in enumerate(observation):
            # Increment for passing
            if o['sticky_actions'][6]:  # Supposing that '6' corresponds to 'Pass'
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]
            
            # Increment for shooting
            if o['sticky_actions'][7]:  # Supposing that '7' corresponds to 'Shot'
                components["shot_reward"][rew_index] = self.shot_reward
                reward[rew_index] += components["shot_reward"][rew_index]
            
            # Increment for dribbling
            if o['sticky_actions'][9]:  # Supposing that '9' corresponds to 'Dribble'
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Increment for sprinting
            if o['sticky_actions'][8]:  # Supposing that '8' corresponds to 'Sprint'
                components["sprint_reward"][rew_index] = self.sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()  # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
