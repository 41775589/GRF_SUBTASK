import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on transitioning skills from defense to attack, emphasizing controlled movement under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_completed = {}  # Tracks completed passes for each agent
        self.ball_control_frames = {}  # Tracks frames where the ball is controlled under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.pass_completed = {}
        self.ball_control_frames = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'pass_completed': self.pass_completed,
            'ball_control_frames': self.ball_control_frames
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_completed = from_pickle['CheckpointRewardWrapper']['pass_completed']
        self.ball_control_frames = from_pickle['CheckpointRewardWrapper']['ball_control_frames']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(),
                      "pass_completion": [0.0] * len(reward),
                      "pressure_control": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = rew
            
            # Reward for completed passes
            passing_actions = [football_action_set.action_short_pass, football_action_set.action_long_pass]
            current_action = o.get('active_action')  # Assuming there's a suitable key in the observation
            if current_action in passing_actions and o['ball_owned_player'] == o['active']:
                self.pass_completed[rew_index] = self.pass_completed.get(rew_index, 0) + 1
                components["pass_completion"][rew_index] = 0.2  # Assuming completion is skillful
            
            # Reward for maintaining ball under pressure
            if o['ball_owned_player'] == o['active'] and self.detect_pressure(o):
                self.ball_control_frames[rew_index] = self.ball_control_frames.get(rew_index, 0) + 1
                if self.ball_control_frames[rew_index] % 10 == 0:  # Every 10 continuous control frames under pressure
                    components["pressure_control"][rew_index] += 0.1
            
            # Update the net reward
            reward[rew_index] = (rew + 
                                 components["pass_completion"][rew_index] +
                                 components["pressure_control"][rew_index])
        
        return reward, components
    
    def detect_pressure(self, observation):
        # Dummy function to determine if a player is under pressure
        # We define "pressure" as having opponent players within a certain radius
        player_pos = observation['left_team'][observation['active']] if observation['ball_owned_team'] == 0 else observation['right_team'][observation['active']]
        opponents = observation['right_team'] if observation['ball_owned_team'] == 0 else observation['left_team']
        
        for opponent in opponents:
            if np.sqrt((opponent[0] - player_pos[0]) ** 2 + (opponent[1] - player_pos[1]) ** 2) < 0.1:  # Arbitrary distance threshold
                return True
        return False
    
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
