import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles during counter-attacks and
    high-pressure situations near our defensive third."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_success_counter = [0, 0]
        self.zone_thresholds = [-0.6, -0.2]  # Define defensive third zones
        self.sliding_tackle_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_success_counter = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sliding_tackle_success_counter': self.sliding_tackle_success_counter,
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_success_counter = from_pickle['CheckpointRewardWrapper']['sliding_tackle_success_counter']
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_tackle_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i in range(len(observation)):
            o = observation[i]
            # Check for defensive third zone and the sliding tackle action
            player_pos = o['left_team'][o['active']]
            team_possession = (o['ball_owned_team'] == 0)
            sliding_action_active = o['sticky_actions'][9] == 1
            
            if team_possession and sliding_action_active and player_pos[0] > self.zone_thresholds[0] and player_pos[0] < self.zone_thresholds[1]:
                components["sliding_tackle_reward"][i] = self.sliding_tackle_reward
                reward[i] += components["sliding_tackle_reward"][i]
                self.sliding_tackle_success_counter[i] += 1
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        if obs:
            self.sticky_actions_counter.fill(0)
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action
        return observation, reward, done, info
