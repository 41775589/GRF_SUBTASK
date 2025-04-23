import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on defensive coordination and ball distribution."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints complexities related to defensive coordination
        self.defensive_checkpoints = 5
        self.defensive_reward = 0.05

        # Define checkpoints for distribution effectiveness 
        self.distribution_checkpoints = 5
        self.distribution_reward = 0.1

        # Reset trackers
        self.defensive_positions = {}
        self.distribution_efficiency = {}

    def reset(self):
        self.defensive_positions = {}
        self.distribution_efficiency = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int) 
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_positions': self.defensive_positions,
            'distribution_efficiency': self.distribution_efficiency
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions = from_pickle['CheckpointRewardWrapper']['defensive_positions']
        self.distribution_efficiency = from_pickle['CheckpointRewardWrapper']['distribution_efficiency']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "distribution_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for maintaining defensive positions
            team_position = o['left_team'] if o['ball_owned_team'] == 1 else o['right_team']
            for player_position in team_position:
                if np.linalg.norm(player_position - o['ball']) < 0.2:  # assuming close to the ball is a good defensive position
                    if rew_index not in self.defensive_positions:
                        components["defensive_reward"][rew_index] = self.defensive_reward
                        self.defensive_positions[rew_index] = 1

            # Reward for effective ball distribution
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                pass_success = np.sum(o['sticky_actions'][7:10])  # assuming actions 7-9 relate to passing/distribution
                if pass_success and rew_index not in self.distribution_efficiency:
                    components["distribution_reward"][rew_index] = self.distribution_reward
                    self.distribution_efficiency[rew_index] = 1
            
            # Calculate the actual reward by summing
            reward[rew_index] += components["defensive_reward"][rew_index] + components["distribution_reward"][rew_index]

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
