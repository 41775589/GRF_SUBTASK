import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides specific rewards to train a goalkeeper:
    Includes rewards for successful shot stopping, rapid decision making under pressure, 
    and effective communication with defenders.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_performance_multiplier = 2.0
        self.communication_bonus = 0.5
        self.slow_penalty = -0.1

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = state['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_performance": [0.0] * len(reward),
                      "communication_bonus": [0.0] * len(reward),
                      "slow_penalty": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]
            
            # When the goalkeeper saves the shot
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] == 0:
                components['goalkeeper_performance'][rew_index] += self.goalkeeper_performance_multiplier
                reward[rew_index] += components['goalkeeper_performance'][rew_index]
            
            # Promote good communication
            if o['ball_owned_team'] == 0 and any(o['left_team_direction']):
                components['communication_bonus'][rew_index] += self.communication_bonus
                reward[rew_index] += components['communication_bonus'][rew_index]
            
            # Penalty for slow play
            if o['steps_left'] > 0 and (self.sticky_actions_counter.sum() == 0):
                components['slow_penalty'][rew_index] += self.slow_penalty
                reward[rew_index] += components['slow_penalty'][rew_index]
            
            # Track sticky actions usage
            for i, action in enumerate(o['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
