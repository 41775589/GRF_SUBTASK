import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward for precise 'Stop-Dribble' handling under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sticky_action_id = 9  # Assuming 'action_dribble' is the 9th action in sticky_actions
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # Reward modifications here
        components = {"base_score_reward": reward.copy(), 
                      "stop_dribble_under_pressure_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        # Iterate over both observations for multi-agent setups
        for i, obs in enumerate(observation):
            # Check if player is under pressure (simple proxy: close to any opponent)
            player_pos = obs['left_team'][obs['active']]
            opponents_pos = obs['right_team']
            min_distance = min(np.linalg.norm(player_pos - opp) for opp in opponents_pos)
            
            # Reward for executing dribble while being closely marked
            if obs['sticky_actions'][self._sticky_action_id] == 1 and min_distance < 0.1:
                # Increasing the reward if under pressure and executing a 'Stop-Dribble'
                pressure_reward = 0.1 * (0.1 - min_distance)  # More reward the closer an opponent is
                components["stop_dribble_under_pressure_reward"][i] += pressure_reward 
                reward[i] += pressure_reward
        
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
