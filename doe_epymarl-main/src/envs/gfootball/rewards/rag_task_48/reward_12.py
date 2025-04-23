import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that enhances reward signals for successful high passes from midfield """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
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
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if we are in a state conducive to midfield high passes
            if 'ball' in o and 'active' in o:
                ball_pos = o['ball']
                active_player_pos = o['right_team'][o['active']] if o['right_team_active'][o['active']] else o['left_team'][o['active']]
                
                # Determine if active player is in midfield and if the ball is currently controlled by the player's team
                midfield_zone = (-0.2 < ball_pos[0] < 0.2)
                controlled_by_team = (o['ball_owned_team'] == 1) if o['right_team_active'][o['active']] else (o['ball_owned_team'] == 0)
                
                if midfield_zone and controlled_by_team and self.check_high_pass(o):
                    # Reward successful high passes from midfield that lead to approaches toward the goal
                    components["high_pass_reward"][rew_index] = 0.5
                    reward[rew_index] += components["high_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.update_sticky_actions(observation)
        return observation, reward, done, info

    def check_high_pass(self, observation):
        """ Check conditions that would qualify a play as a successful high pass resulting in scoring chances """
        if 'ball_direction' in observation and 'steps_left' in observation:
            # Simple heuristic: High passes generally involve a vertical (y-axis) movement
            considerable_y_movement = abs(observation['ball_direction'][1]) > 0.1
            target_approach = observation['ball_direction'][0] > 0 if observation['right_team_active'][observation['active']] else observation['ball_direction'][0] < 0
            return considerable_y_movement and target_approach
        return False

    def update_sticky_actions(self, obs):
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
