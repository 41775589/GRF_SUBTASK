import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on rewarding good positioning and high pass usage by wide midfielders to expand play and
    aid in lateral transitions. This aims to stretch the opposition's defense and create space effectively.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track the use of sticky actions
        
    def reset(self):
        """
        Reset sticky actions counter when the environment is reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Save wrapper state with the main environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.copy()}
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        """
        Restore wrapper state from saved state.
        """
        state_info = self.env.set_state(from_pickle)
        wrapper_info = state_info.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = wrapper_info.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state_info

    def reward(self, reward):
        """
        Modify rewards based on the agent's effectiveness in executing wide midfielder responsibilities
        including high passing and positioning to open gameplay.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "high_pass_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]  # Considering only x, y coordinates
            
            # Reward for lateral movement: check if on wide midfield and moves laterally toward center
            if abs(player_pos[1]) > 0.25:  # Assuming midfield width
                components["positioning_reward"][rew_index] += 0.01  # Small reward for being in position
                
            # Check high pass usage by sticky action index for high pass
            if o['sticky_actions'][7] == 1:  # Assuming index 7 represents high pass
                self.sticky_actions_counter[7] += 1
                x_distance = abs(ball_pos[0] - player_pos[0])
                components["high_pass_reward"][rew_index] += 0.05 * x_distance  # Reward based on distance ball travelled in x
            
            # Combine rewards
            reward[rew_index] += components["positioning_reward"][rew_index] + components["high_pass_reward"][rew_index]
        
        return reward, components
    
    def step(self, action):
        """
        Execute a step with the wrapped environment and enrich the supplied 'info' dictionary with
        component-wise reward data for reporting or debugging.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
