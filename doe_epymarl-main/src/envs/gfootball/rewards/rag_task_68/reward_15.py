import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to emphasize offensive strategies in football game simulation."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.3
        self.passing_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        self.sticky_actions_counter = np.asarray(counter, dtype=int)
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
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Component for dribbling (using dribble action effectively)
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0 and o['right_team_active'].any():  # Assume '0' is always our team
                distance = np.linalg.norm(o['right_team'][o['ball_owned_player']] - o['left_team'], axis=1).min()
                if distance < 0.1:  # Effectively evading close opponents
                    components['dribbling_reward'][rew_index] = self.dribbling_reward
                    reward[rew_index] += components['dribbling_reward'][rew_index]
            
            # Component for effective shooting
            if o['game_mode'] in (2, 6) and o['ball_owned_team'] == 0:  # Considering FreeKick and Penalty modes
                goal_distance = abs(o['ball'][0] - 1)  # Goal at x=1
                if goal_distance < 0.15:  # Near the goal
                    components['shooting_reward'][rew_index] = self.shooting_reward
                    reward[rew_index] += components['shooting_reward'][rew_index]
            
            # Component for passing
            if np.any(o['sticky_actions'][7:9]):  # Bottom_left or bottom (assumed to be long or high passes)
                if o['ball_owned_team'] == 0:
                    pass_quality = np.linalg.norm(o['ball_direction'][:2])  # Simplicity assumed pass quality based on speed vector magnitude
                    if pass_quality > 0.5:
                        components['passing_reward'][rew_index] = self.passing_reward
                        reward[rew_index] += components['passing_reward'][rew_index]

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
