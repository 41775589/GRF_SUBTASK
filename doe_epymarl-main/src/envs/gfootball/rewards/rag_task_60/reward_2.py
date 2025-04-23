import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive player adaptation through precise stopping
    and starting movements, focusing on rapid transitions to counter offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for rapid transitions
        self.thresh_start = 0.1  # threshold for rapid start
        self.thresh_stop = 0.1   # threshold for rapid stop

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
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        transition_coefficient = 0.05  # Reward scaling factor for effective transitions

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_motion = np.linalg.norm(o['active_player_direction'])
            sticky_actions = o['sticky_actions']
            rapid_start = player_motion > self.thresh_start and np.sum(sticky_actions) == 0
            rapid_stop = player_motion < self.thresh_stop and np.sum(sticky_actions) > 0

            if rapid_start or rapid_stop:
                # Reward defensive transitions effectively to react rapidly to plays
                components["transition_reward"][rew_index] = transition_coefficient
                reward[rew_index] += components["transition_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
