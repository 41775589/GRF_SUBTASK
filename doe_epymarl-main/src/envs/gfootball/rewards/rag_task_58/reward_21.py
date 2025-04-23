import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defensive coordination reward focusing on ball interception,
    defense stability, and transitioning to attack after gaining possession.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_intercepted = False
        self._defense_stability_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_intercepted = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['ball_intercepted'] = self._ball_intercepted
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self._ball_intercepted = from_pickle['ball_intercepted']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'defense_reward': [0.0] * len(reward)}

        # If observation is None, return unmodified reward.
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0: # Assuming 0 is the controlled team
                # Encourage maintaining ball possession.
                if not self._ball_intercepted:
                    components['defense_reward'][i] += self._defense_stability_reward
                    reward[i] += components['defense_reward'][i]
                    self._ball_intercepted = True
            else:
                self._ball_intercepted = False

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
