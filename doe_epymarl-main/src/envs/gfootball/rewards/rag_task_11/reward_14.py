import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive maneuvers and precision finishing control."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checkpoints = 0
        self.finish_positions = []
        self.pass_reward = 0.05
        self.finish_reward = 0.1
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checkpoints = 0
        self.finish_positions = []
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'pass_checkpoints': self.pass_checkpoints,
            'finish_positions': self.finish_positions
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_checkpoints = from_pickle['CheckpointRewardWrapper']['pass_checkpoints']
        self.finish_positions = from_pickle['CheckpointRewardWrapper']['finish_positions']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "finish_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful passes in the offensive play
            if o['ball_owned_team'] == 1 and o['game_mode'] == 0:  # Assuming player's team is the right side (index 1)
                if (o['ball_owned_player'] != -1 and 
                    o['ball_owned_player'] not in self.finish_positions):
                    self.pass_checkpoints += 1
                    components["pass_reward"][rew_index] = self.pass_reward
                    self.finish_positions.append(o['ball_owned_player'])
            
            # Reward for getting the ball into the finish zone (close to opponent's goal)
            x_finish_threshold = 0.8  # x-coordinate close to the opponent's goal
            if o['ball'][0] > x_finish_threshold and o['ball_owned_team'] == 1:
                components["finish_reward"][rew_index] = self.finish_reward
            
            reward[rew_index] += components["pass_reward"][rew_index] + components["finish_reward"][rew_index]
        
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
