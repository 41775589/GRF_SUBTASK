import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.2

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
        components = {
            "base_score_reward": reward.copy(),
            "defensive_play_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            action_defensive_benefit = self.calculate_defensive_reward(o)
            components["defensive_play_reward"][rew_index] = action_defensive_benefit
            reward[rew_index] += action_defensive_benefit
            
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

    def calculate_defensive_reward(self, observation):
        """ Calculate the reward for defensive actions based on the ball possession and player position. 
        - Players get a reward for intercepting the ball or being positioned effectively in a defensive scenario.
        """
        reward = 0
        if observation['ball_owned_team'] == 1:  # Opponent has the ball
            # Use Euclidean distance to ball from closest defender
            ball_position = observation['ball'][:2]
            team_positions = observation['left_team'] if observation['ball_owned_team'] == 1 else observation['right_team']
            distances = np.linalg.norm(team_positions - ball_position, axis=1)
            closest_defender_distance = np.min(distances)
            # Award based on proximity to the ball, inversely proportional, encouraging closer defense.
            if closest_defender_distance < 0.1:
                reward += self.interception_reward
      
        return reward
