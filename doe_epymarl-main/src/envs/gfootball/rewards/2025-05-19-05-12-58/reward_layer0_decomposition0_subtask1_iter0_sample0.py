import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive tactics and efficiency."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """Resets the environment and the necessary attributes for the reward computation."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encapsulates the state with additional wrapper-specific data."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state along with the wrapper-specific properties."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Adds a defensive proficiency based reward logic to the base reward."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Increase the reward for successful defensive actions
            if o['game_mode'] in [2, 3, 4, 6]:  # Defensive game modes: Goal Kick, Free Kick, Corner, Penalty
                components["defensive_reward"][rew_index] += 0.5
                reward[rew_index] += components["defensive_reward"][rew_index]
            
            # Additional rewards for maintaining possession safely in defensive third
            if o['ball_owned_team'] == 0 and abs(o['ball'][0]) > 0.5:  # Assuming that left side is defensive
                components["defensive_reward"][rew_index] += 0.3
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Coordinate movements in the defensive half
            for player_pos in o['left_team']:
                if player_pos[0] < -0.5:  # Consider defensive half to be left side < -0.5
                    components["defensive_reward"][rew_index] += 0.1
                    reward[rew_index] += components["defensive_reward"][rew_index]
                    break
        
        return reward, components

    def step(self, action):
        """Performs an environment step and augments output with reward component data."""
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
