import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a nuanced defensive skill training reward."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define key zones of defending effectively and transitioning quickly
        self.defensive_zones = [
            (-1, -0.42, -0.7),  # Zones approaching our goal area, left, right and center
            (-1, -0.42, -0.15),
            (-1, -0.42, 0.15),
            (-1, -0.42, 0.7)
        ]
        self.transition_reward = 0.1
        self.defensive_reward = 0.5
        self.defensive_engage_distance = 0.1  # Distance threshold for engaging in defense
        self.transition_engage_distance = 0.5  # Distance threshold for beginning transitions

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
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check defensive positioning
            if o['ball_owned_team'] in [0, 1]:  # Assuming the opponent is the left team
                ball_pos = np.array(o['ball'][:2])
                if o['ball_owned_team'] == 0:  # Opponent has the ball
                    for zone in self.defensive_zones:
                        zone_pos = np.array([zone[0], zone[2]])
                        if np.linalg.norm(ball_pos - zone_pos) <= self.defensive_engage_distance:
                            components["defensive_reward"][rew_index] += self.defensive_reward
                            
            # Check for successful transition to attack
            if o['ball_owned_team'] == 1:  # Our team has the ball
                own_pos = np.array(o['left_team'][o['active']][:2])
                enemy_goal = np.array([1, 0])  # Opponent's goal position
                if np.linalg.norm(own_pos - enemy_goal) <= self.transition_engage_distance:
                    components["transition_reward"][rew_index] += self.transition_reward
                    
            reward[rew_index] += components["defensive_reward"][rew_index] + components["transition_reward"][rew_index]
            
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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
