import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on advanced dribbling and sprint usage."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Extract observations from the wrapped environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]

            # Dribbling reward increases with controlled dribbling around defenders
            if player_obs['sticky_actions'][9]:  # Dribbling action is active
                components["dribbling_reward"][rew_index] = 0.1
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Sprint reward boosts reward if used strategically while dribbling
            if player_obs['sticky_actions'][8] and player_obs['sticky_actions'][9]:  # Sprint + dribble
                components["sprint_reward"][rew_index] = 0.1
                reward[rew_index] += components["sprint_reward"][rew_index]

            # Update sticky actions counter
            self.sticky_actions_counter += player_obs['sticky_actions']

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
