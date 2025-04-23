import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper that encourages advanced ball control and passing under pressure, focusing on Short Pass,
    High Pass, and Long Pass in tight game situations. The reward function provides incentive for successful passing
    under these conditions, encouraging skillful player movement and interaction.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_multiplier = 0.1  # Reward multiplier for each successful pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        """
        Modifies the reward based on the quality and conditions of passing under pressure.
        Short Pass (5), High Pass (6), Long Pass (7) are considered here.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):  # typically there are two players in observations
            o = observation[i]
            
            # Check if the active player's team owns the ball
            if o['ball_owned_team'] == o['ball_owned_player']:
                pass_type = self.current_pass_type(o['sticky_actions'])
                pressure = self.calculate_pressure(o)
                
                # If under pressure and perform one of the pass actions, grant additional reward
                if pass_type and pressure:
                    components['passing_reward'][i] += self.pass_reward_multiplier
                    reward[i] += components['passing_reward'][i]

        return reward, components

    def current_pass_type(self, sticky_actions):
        """
        Returns True if one of the passing actions is being performed.
        Indices in sticky actions for passes are considered as:
        5 - Short Pass
        6 - High Pass
        7 - Long Pass
        """
        return sticky_actions[5] == 1 or sticky_actions[6] == 1 or sticky_actions[7] == 1
    
    def calculate_pressure(self, observation):
        """
        Determine if the player is under pressure from opponents.
        Simplified as being very close to an opponent player.
        """
        player_pos = observation['right_team'][observation['active']] if observation['ball_owned_team'] == 1 else observation['left_team'][observation['active']]
        opponents_pos = observation['left_team'] if observation['ball_owned_team'] == 1 else observation['right_team']
        
        # Calculate distance to all opponents and consider under pressure if any is closer than a threshold
        pressure_threshold = 0.1  # Arbitrary threshold for being under pressure
        under_pressure = any(np.linalg.norm(player_pos - opp_pos) < pressure_threshold for opp_pos in opponents_pos)
        return under_pressure

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
