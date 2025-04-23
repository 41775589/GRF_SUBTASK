import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for advancing ball control and passing under pressure."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward = 0.1
        self.control_under_pressure_reward = 0.2
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'pass_quality_reward': [0.0] * len(reward),
                      'control_under_pressure_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Check through each agent's observation
        for i in range(len(reward)):
            obs = observation[i]
            base_reward = reward[i]
            
            # Encourage playing the ball while close to opponents
            if obs['ball_owned_team'] == 0:  # Assuming team 0 is the controlled team
                for opponent in obs['right_team']:
                    distance = np.linalg.norm(obs['ball'][:2] - opponent)
                    # Reward for playing under pressure if close to opponents
                    if distance < 0.1: # arbitrary close distance threshold
                        components['control_under_pressure_reward'][i] += self.control_under_pressure_reward
                        reward[i] += components['control_under_pressure_reward'][i]
            
            # Encourage successful passes
            if obs['game_mode'] in {1, 4, 6}: # Game modes that involve passing (KickOff, Corner, Penalty)
                # Successful pass
                if obs['ball_owned_player'] in obs['left_team_designated'] and obs['ball_owned_team'] == 0:
                    components['pass_quality_reward'][i] += self.pass_quality_reward
                    reward[i] += components['pass_quality_reward'][i]
                            
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Add component values and final reward to info
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f'sticky_actions_{i}'] = action_active
        
        return observation, reward, done, info
