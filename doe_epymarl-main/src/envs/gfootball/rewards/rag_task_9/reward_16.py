import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focusing on offensive skills
    like passing, shooting, and dribbling to create scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters for tracking progress through specific actions
        self.pass_reward_increase = 0.1
        self.shot_reward_increase = 0.2
        self.dribbling_reward_increase = 0.05
        self.sprint_reward_increase = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "sprinting_reward": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            o = observation[i]
            # Reward for action is based on passing (actions 1 or 2 denote short and long passes)
            if o['sticky_actions'][1] == 1 or o['sticky_actions'][2] == 1:
                components["passing_reward"][i] = self.pass_reward_increase
            # Reward for shooting (action 8 indicates a shot at goal)
            if o['sticky_actions'][8] == 1:
                components["shooting_reward"][i] = self.shot_reward_increase
            # Reward for dribbling (action 9 indicates dribbling)
            if o['sticky_actions'][9] == 1:
                components["dribbling_reward"][i] = self.dribbling_reward_increase
            # Reward for sprinting (action 7 indicates sprinting)
            if o['sticky_actions'][7] == 1:
                components["sprinting_reward"][i] = self.sprint_reward_increase

            reward[i] += (components["passing_reward"][i] + 
                          components["shooting_reward"][i] +
                          components["dribbling_reward"][i] +
                          components["sprinting_reward"][i])

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
