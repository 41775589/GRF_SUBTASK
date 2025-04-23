import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for wingers crossing and sprint performance optimizations.
    Rewards are specifically designed to incentivize players to practice high-speed dribbling on wings and
    perform successful crosses, crucial skills for wingers.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions usage
        self._crossing_regions = 5  # Number of specific regions on the wings
        self._crossing_bonus = 0.1  # Bonus for each valid cross

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions counter
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Encourage players reaching the wings with the ball and attempting crosses
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0.5:  # Right half
                if abs(o['ball'][1]) > 0.3:  # Y position near wing areas
                    if o['sticky_actions'][7] or o['sticky_actions'][1]:  # Is crossing
                        components["crossing_bonus"][rew_index] = self._crossing_bonus
                        reward[rew_index] += components["crossing_bonus"][rew_index]
            elif o['ball_owned_team'] == 0 and o['ball'][0] < -0.5:  # Left half
                if abs(o['ball'][1]) > 0.3:  # Y position near wing areas
                    if o['sticky_actions'][7] or o['sticky_actions'][1]:  # Is crossing
                        components["crossing_bonus"][rew_index] = self._crossing_bonus
                        reward[rew_index] += components["crossing_bonus"][rew_index]

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
        return observation, reward, done, info
