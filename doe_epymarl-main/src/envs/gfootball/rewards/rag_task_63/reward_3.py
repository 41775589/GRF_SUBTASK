import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on training a goalkeeper in football."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position = None
        self.initial_distance = None

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.goalkeeper_position = None
        self.initial_distance = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'goalkeeper_position': self.goalkeeper_position,
            'initial_distance': self.initial_distance
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = state_info['sticky_actions_counter']
        self.goalkeeper_position = state_info['goalkeeper_position']
        self.initial_distance = state_info['initial_distance']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        reward_components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, reward_components

        for i, o in enumerate(observation):
            if 'left_team_roles' in o and 'ball' in o:
                if self.goalkeeper_position is None and o['left_team_roles'][o['active']] == 0:
                    self.goalkeeper_position = o['left_team'][o['active']]
                    self.initial_distance = np.linalg.norm(o['ball'][:2] - self.goalkeeper_position)
                
                current_distance = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']])
                
                # Reward goalkeeper for reducing the distance to the ball
                if current_distance < self.initial_distance:
                    reward_components['positioning_reward'][i] = 0.1 * (self.initial_distance - current_distance)
                    reward[i] += reward_components['positioning_reward'][i]

                # Reward for effective communication (simulated by low action frequency)
                if np.sum(self.sticky_actions_counter) < 5:
                    reward_components['communication_reward'][i] = 0.05
                    reward[i] += reward_components['communication_reward'][i]

        return reward, reward_components

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
