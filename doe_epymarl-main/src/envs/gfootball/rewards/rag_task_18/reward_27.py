import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes on controlled pace and transition efficiency 
    by the central midfield players.
    """
    def __init__(self, env):
        super().__init__(env)
        self.central_midfield_control = 0.05  # Reward for effective transitions by central midfielders
        self.pace_management = 0.03  # Reward for maintaining an optimal pace
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the wrapper state and the environment.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state for serialization.
        """
        to_pickle['CheckpointRewardWrapper'] = dict(sticky_actions_counter=self.sticky_actions_counter.tolist())
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore state from deserialization.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle
    
    def reward(self, reward):
        """
        Modify the reward function to incentivize tactical synergy and effective pace management by
        central midfielders. 
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "central_midfield_control": [0.0] * len(reward),
                      "pace_management": [0.0] * len(reward)}

        assert len(reward) == len(observation)
        
        for reward_index in range(len(reward)):
            o = observation[reward_index]
            
            # Reward for central midfield players performing effective transitions 
            if o['active'] in o['left_team_roles'] and o['left_team_roles'][o['active']] == 5: # 5 represents central midfield
                ball_direction_magnitude = np.linalg.norm(o['ball_direction'][:2])
                if ball_direction_magnitude < 0.1:  # Assuming a lower magnitude leads to better control
                    components["central_midfield_control"][reward_index] = self.central_midfield_control
                    reward[reward_index] += components["central_midfield_control"][reward_index]
            
            # Rewarding pace management by evaluating the speed consistency
            player_speed = np.linalg.norm(o['left_team_direction'][o['active'], :])
            if 0.03 <= player_speed <= 0.1:  # Arbitrarily assumed optimal speed range for good pace
                components["pace_management"][reward_index] = self.pace_management
                reward[reward_index] += components["pace_management"][reward_index]

        return reward, components
        
    def step(self, action):
        """
        Steps through the environment, apply reward modifications, and return state and modifed reward.
        """
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
