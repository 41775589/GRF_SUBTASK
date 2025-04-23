import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward related to goalkeeper coordination
    and strategic ball clearing to specific outfield players.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position = None
        self.last_high_pressure_state = False
        self.high_pressure_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position = None
        self.last_high_pressure_state = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'goalkeeper_position': self.goalkeeper_position,
            'last_high_pressure_state': self.last_high_pressure_state
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_position = from_pickle['CheckpointRewardWrapper']['goalkeeper_position']
        self.last_high_pressure_state = from_pickle['CheckpointRewardWrapper']['last_high_pressure_state']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_coordination_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        # Calculate rewards related to goalkeeper coordination
        for i in range(len(reward)):
            o = observation[i]
            if 'left_team_roles' in o and 'right_team_roles' in o:
                roles_left = o['left_team_roles']
                roles_right = o['right_team_roles']
                
                # Identify the goalkeeper (role index 0) and store position
                if i == 0:  # Assuming left team is our focus
                    goalkeeper_index = np.where(roles_left == 0)[0]
                    if goalkeeper_index.size > 0:
                        goalkeeper_position = o['left_team'][goalkeeper_index]
                        self.goalkeeper_position = goalkeeper_position
                        
                # Calculate pressures near the goalkeeper
                own_team = o['left_team'] if i == 0 else o['right_team']
                opponent_team = o['right_team'] if i == 0 else o['left_team']
                if self.goalkeeper_position is not None:
                    distances = np.linalg.norm(opponent_team - self.goalkeeper_position, axis=1)
                    high_pressure = np.any(distances < 0.1)  # High pressure defined as opponents very close

                    # Check transition from normal to high pressure state
                    if high_pressure and not self.last_high_pressure_state:
                        components["goalkeeper_coordination_reward"][i] = self.high_pressure_reward
                        reward[i] += components["goalkeeper_coordination_reward"][i]
                        self.last_high_pressure_state = True
                    elif not high_pressure:
                        self.last_high_pressure_state = False
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
