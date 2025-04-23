import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on specialized goalkeeper training including shot-stopping,
    quick reflexes, and initiating counter-attacks with accurate passes."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_on_target = {0: 0, 1: 0}
        self.passes_completed = {0: 0, 1: 0}
        self.reflexes_enhanced = {0: 0, 1: 0}
        self.shot_stopping_reward = 1.0
        self.passing_reward = 0.5
        self.reflex_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_on_target = {0: 0, 1: 0}
        self.passes_completed = {0: 0, 1: 0}
        self.reflexes_enhanced = {0: 0, 1: 0}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shots_on_target': self.shots_on_target,
            'passes_completed': self.passes_completed,
            'reflexes_enhanced': self.reflexes_enhanced
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        data = from_pickle['CheckpointRewardWrapper']
        self.shots_on_target = data['shots_on_target']
        self.passes_completed = data['passes_completed']
        self.reflexes_enhanced = data['reflexes_enhanced']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shot_stopping": [0.0, 0.0], "passing": [0.0, 0.0], "reflex": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalkeeper_index = np.argmin([p[1] for p in o['left_team']])  # assuming goalkeeper is closest to goal line
            
            # Reward shot stopping
            if o['ball_owned_team'] == 1 and o['active'] == goalkeeper_index and o['game_mode'] in [2, 6]:  # Ball close or in penalty mode
                components['shot_stopping'][rew_index] = self.shot_stopping_reward
                self.shots_on_target[rew_index] += 1
                
            # Reward reflex enhancement
            ball_speed = np.linalg.norm(o['ball_direction'])
            if o['ball_owned_team'] == 1 and ball_speed > 0.05 and o['active'] == goalkeeper_index:
                components['reflex'][rew_index] = self.reflex_reward
                self.reflexes_enhanced[rew_index] += 1
                
            # Reward accurate passes by goalkeeper
            if o['designated'] == goalkeeper_index and o['ball_owned_player'] == goalkeeper_index and o['sticky_actions'][4] == 1:  # Assuming action 4 is 'kick'
                components['passing'][rew_index] = self.passing_reward
                self.passes_completed[rew_index] += 1

            reward[rew_index] += sum(components.values())
            
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
