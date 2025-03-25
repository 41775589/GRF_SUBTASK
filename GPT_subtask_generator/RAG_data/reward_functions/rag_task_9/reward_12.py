import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense, task-specific reward focused on offensive actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.1
        self.shot_reward = 0.2
        self.dribble_reward = 0.05
        self.sprint_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        # Initialize component rewards
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        # Apply rewards based on the specific offensive actions taken
        for idx in range(len(reward)):
            agent_obs = observation[idx]
            sticky_actions = agent_obs['sticky_actions']
            
            # Rewards for pass, shot, dribble, sprint actions
            if sticky_actions[6]:  # Assuming this is the index for "Short Pass"
                components["pass_reward"][idx] = self.pass_reward
                reward[idx] += components["pass_reward"][idx]

            if sticky_actions[7]:  # Assuming this is the index for "Long Pass"
                components["pass_reward"][idx] += self.pass_reward
                reward[idx] += components["pass_reward"][idx]

            if sticky_actions[9]:  # Assuming this is the index for "Shot"
                components["shot_reward"][idx] = self.shot_reward
                reward[idx] += components["shot_reward"][idx]

            if sticky_actions[8]:  # Assuming this is the index for "Dribble"
                components["dribble_reward"][idx] = self.dribble_reward
                reward[idx] += components["dribble_reward"][idx]

            if sticky_actions[0]:  # Assuming this is the index for "Sprint"
                components["sprint_reward"][idx] = self.sprint_reward
                reward[idx] += components["sprint_reward"][idx]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
