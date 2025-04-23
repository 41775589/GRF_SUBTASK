import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on ball control and effective passing 
    under pressure, particularly emphasizing the quality of Short Pass, High Pass, 
    and Long Pass in tight situations.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_success_reward = 0.3
        self.control_under_pressure_reward = 0.2
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = self.env.get_state(to_pickle)
        state_data['sticky_actions_counter'] = self.sticky_actions_counter
        return state_data

    def set_state(self, state):
        state_data = self.env.set_state(state)
        self.sticky_actions_counter = state_data['sticky_actions_counter']
        return state_data

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0] * len(reward)}
        
        if not observation:
            return reward, components
        
        assert len(reward) == len(observation), "Reward and observation lengths must match"
        
        for i in range(len(reward)):
            obs = observation[i]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:  # ball is owned by the team
                # Check the passing ability under pressure
                if 'sticky_actions' in obs:
                    # Checking for (Short Pass, High Pass, Long Pass)
                    is_pass_initiated = obs['sticky_actions'][5] or obs['sticky_actions'][7] or obs['sticky_actions'][9]
                    if is_pass_initiated:
                        components['pass_success_reward'][i] = self.pass_success_reward
                        reward[i] += self.pass_success_reward
                
                # Check control under game pressure scenarios
                player_pos = obs['left_team'][obs['active']]
                opponent_dists = np.linalg.norm(obs['right_team'] - player_pos, axis=1)
                # Pressure is deemed as at least 3 opponents within a range of 0.1 units
                if np.sum(opponent_dists < 0.1) >= 3:
                    components['control_under_pressure_reward'][i] = self.control_under_pressure_reward
                    reward[i] += self.control_under_pressure_reward
            
        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        current_obs = self.env.unwrapped.observation()
        for agent_obs in current_obs:
            for j, action_state in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{j}'] = action_state
        
        return obs, reward, done, info
