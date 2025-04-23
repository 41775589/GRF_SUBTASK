import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that extends the environment to increase learning on defensive maneuvers, specifically focusing on 
    standing and sliding tackles without causing fouls, during diverse gameplay scenarios.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To track action frequencies

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
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            # Check if a tackle action was performed
            if o['sticky_actions'][6] == 1 or o['sticky_actions'][7] == 1:  # assuming indices 6,7 are tackle actions
                # Safeguarding against fouls, motivate clean tackles
                if o['game_mode'] == 6:  # Assuming game_mode 6 indicates a foul
                    reward[i] -= 0.5  # Penalize for causing a foul during a tackle
                else:
                    reward[i] += 0.1  # Reward for a proper tackle
                    
            components['sticky_actions_counts'] = self.sticky_actions_counter.tolist()
            
            # Update action frequency count
            self.sticky_actions_counter += o['sticky_actions']
        
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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        
        return observation, reward, done, info
