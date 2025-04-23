import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper focused on enhancing team synergy during possession changes.
    It rewards players for precise positioning and strategic timing during changes
    in possession, encouraging synchronization between offensive and defensive maneuvers.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reset()
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = -1
        self.possession_change_locs = []
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_possession'] = self.previous_possession
        to_pickle['possession_change_locs'] = self.possession_change_locs
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.previous_possession = from_pickle['previous_possession']
        self.possession_change_locs = from_pickle['possession_change_locs']
        return from_pickle

    def reward(self, reward):
        """
        Enhances the reward based on strategic player positioning during possession changes.
        This includes rewards for intercepting the ball and repositioning effectively.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}
        
        ball_owned_team = observation['ball_owned_team']
        
        if self.previous_possession != ball_owned_team and ball_owned_team != -1:
            change_location = observation['ball']
            # Reward agents for possession changes at critical locations
            # Assume critical locations are near goals or central areas.
            critical_regions = [(0, 0)]  # The center of the pitch
            for region in critical_regions:
                distance = np.linalg.norm(np.array(region) - np.array(change_location[0:2]))
                if distance < 0.1:
                    components["possession_change_reward"] = [0.5, 0.5]  # Both agents rewarded

            self.possession_change_locs.append(change_location)
            
        self.previous_possession = ball_owned_team
        enhanced_reward = [r + components["possession_change_reward"][i] for i, r in enumerate(reward)]
        
        return enhanced_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
