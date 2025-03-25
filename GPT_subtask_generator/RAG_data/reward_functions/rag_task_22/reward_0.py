import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focused on enhancing defensive gameplay through sprint-focused rewards, 
    in an effort to encourage quicker and more effective repositioning of players to adapt 
    to game dynamics.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_usage_rewards = np.zeros(2, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_reward_coefficient = 0.05

    def reset(self):
        self.sprint_usage_rewards = np.zeros(2, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_usage_rewards'] = self.sprint_usage_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_usage_rewards = from_pickle['sprint_usage_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        reward_components = {"base_score_reward": reward.copy(),
                             "sprint_usage_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            agent_obs = observation[i]
            if agent_obs['sticky_actions'][8]:  # Index 8 corresponds to 'sprint'
                # Reward agents for making use of the sprint action effectively.
                self.sprint_usage_rewards[i] += 1
                reward_components["sprint_usage_reward"][i] = self.sprint_reward_coefficient

            reward[i] += self.sprint_reward_coefficient * self.sprint_usage_rewards[i]

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
