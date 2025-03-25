import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward based on agents' sprint usage
    to improve defensive movements across the field.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sprint_rewards = {}
        self.total_sprints = 0
        self.sprint_reward_increment = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and reward parameters."""
        self.sprint_rewards = {}
        self.total_sprints = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encodes the state of the environment and reward wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sprint_rewards': self.sprint_rewards,
            'total_sprints': self.total_sprints
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Decodes and loads the state of the environment and reward wrapper
        from previously saved state.
        """
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.sprint_rewards = state_data['sprint_rewards']
        self.total_sprints = state_data['total_sprints']
        return from_pickle

    def reward(self, reward):
        """
        Augments the reward based on agent's sprint usage to encourage faster
        positioning across the field for defensive purposes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, obs in enumerate(observation):
            if obs['sticky_actions'][8]:  # Index for sprint action
                self.total_sprints += 1
                self.sprint_rewards[rew_index] = self.sprint_rewards.get(rew_index, 0) + self.sprint_reward_increment
                reward[rew_index] += self.sprint_rewards[rew_index]
                components["sprint_reward"][rew_index] = self.sprint_rewards[rew_index]

        return reward, components

    def step(self, action):
        """
        Steps through environment, processes actions, and returns observation with
        augmented rewards.
        """
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
