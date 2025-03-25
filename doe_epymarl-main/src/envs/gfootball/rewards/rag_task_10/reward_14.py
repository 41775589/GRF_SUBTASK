import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive reward focusing on defensive skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions = {
            'slide': 0.2,  # reward for sliding (typically a defensive move)
            'intercept': 0.3,  # reward for intercepting passes
            'mark_player': 0.1,  # reward for marking an opposition player effectively
            'tackle': 0.25  # reward for tackling an opponent with the ball
        }

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
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            defensive_reward = 0
            
            # Check if there are defensive actions being performed and reward them
            if 'slide' in o['sticky_actions'] and o['sticky_actions']['slide']:
                defensive_reward += self.defensive_actions['slide']
            if 'intercept' in o['sticky_actions'] and o['sticky_actions']['intercept']:
                defensive_reward += self.defensive_actions['intercept']
            if 'mark_player' in o['sticky_actions'] and o['sticky_actions']['mark_player']:
                defensive_reward += self.defensive_actions['mark_player']
            if 'tackle' in o['sticky_actions'] and o['sticky_actions']['tackle']:
                defensive_reward += self.defensive_actions['tackle']
            
            reward[rew_index] += defensive_reward
            if 'reward_components' not in components:
                components['reward_components'] = []
            components['reward_components'].append(defensive_reward)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
