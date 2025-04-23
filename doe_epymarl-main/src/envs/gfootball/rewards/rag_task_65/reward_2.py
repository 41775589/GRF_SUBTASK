import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a scenario-based reward for shooting and passing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize variables to track shooting and passing scenarios
        self._shooting_positions = np.linspace(-1, 1, 5)
        self._passing_positions = np.linspace(-1, 1, 5)
        self._shooting_reward = 0.1
        self._passing_reward = 0.05
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize reward components
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Evaluate each agent in the environment
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Adding rewards for scenarios involving shooting
            if o['ball_owned_player'] == o['active'] and o['ball'][0] in self._shooting_positions:
                components["shooting_reward"][rew_index] = self._shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Adding rewards for scenarios involving passing
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and o['ball'][0] in self._passing_positions:
                components["passing_reward"][rew_index] = self._passing_reward
                reward[rew_index] += components["passing_reward"][rew_index]

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
