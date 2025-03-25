import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective sprint usage which improves defensive coverage."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._distance_threshold = 0.1
        self._sprint_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_usage_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:
                if obs['sticky_actions'][8] == 1:  # Check if sprint action is active
                    distance_to_ball = np.linalg.norm(np.array(obs['left_team'][obs['active']]) - np.array(obs['ball'][:2]))
                    if distance_to_ball < self._distance_threshold:
                        components['sprint_usage_reward'][rew_index] = self._sprint_reward
                        reward[rew_index] += components['sprint_usage_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info['final_reward'] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter[:] = 0
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        return observation, new_reward, done, info
