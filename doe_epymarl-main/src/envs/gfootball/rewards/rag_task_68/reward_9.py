import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on offense strategies: shooting, dribbling, and passing."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for dribbling and shooting attempts
        self.shooting_counter = np.zeros(10, dtype=int)
        self.dribbling_counter = np.zeros(10, dtype=int)
        self.passing_counter = np.zeros(10, dtype=int)
        
        # Reward coefficients
        self.shooting_reward_coefficient = 0.2
        self.dribbling_reward_coefficient = 0.15
        self.passing_reward_coefficient = 0.1
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Resetting limits
        self.shooting_limit = 5
        self.dribbling_limit = 7
        self.passing_limit = 10
    
    def reset(self):
        self.shooting_counter.fill(0)
        self.dribbling_counter.fill(0)
        self.passing_counter.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shooting_counter': self.shooting_counter, 
            'dribbling_counter': self.dribbling_counter,
            'passing_counter': self.passing_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_counter = from_pickle['CheckpointRewardWrapper']['shooting_counter']
        self.dribbling_counter = from_pickle['CheckpointRewardWrapper']['dribbling_counter']
        self.passing_counter = from_pickle['CheckpointRewardWrapper']['passing_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, rew_value in enumerate(reward):
            o = observation[rew_index]
            
            # Shooting reward calculation
            if o['sticky_actions'][9] == 1 and self.shooting_counter[rew_index] < self.shooting_limit:
                self.shooting_counter[rew_index] += 1
                components['shooting_reward'][rew_index] = self.shooting_reward_coefficient
            
            # Dribbling rewarded when player has the ball and uses dribble action
            if o['sticky_actions'][8] == 1 and o['ball_owned_team'] == 0 and \
               o['ball_owned_player'] == o['active'] and self.dribbling_counter[rew_index] < self.dribbling_limit:
                self.dribbling_counter[rew_index] += 1
                components['dribbling_reward'][rew_index] = self.dribbling_reward_coefficient
            
            # Passing reward when a pass changes ball ownership within the team
            if o['sticky_actions'][0] or o['sticky_actions'][4]:  # Assumed indices for pass-related actions
                if (self.env.last_observation is not None and 
                    self.env.last_observation[rew_index]['ball_owned_team'] == o['ball_owned_team'] == 0 and 
                    self.env.last_observation[rew_index]['ball_owned_player'] != o['ball_owned_player'] and 
                    self.passing_counter[rew_index] < self.passing_limit):
                    self.passing_counter[rew_index] += 1
                    components['passing_reward'][rew_index] = self.passing_reward_coefficient
            
            # Aggregate reward
            reward[rew_index] += components['shooting_reward'][rew_index] + \
                                 components['dribbling_reward'][rew_index] + \
                                 components['passing_reward'][rew_index]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
