import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense cooperative reward to encourage midfielders in space creation 
    and strikers in finishing plays in an offensive strategy.
    """

    def __init__(self, env):
        super().__init__(env)
        # Define the number of zones or segments in midfield for cooperative play
        self.midfield_zones = 5
        self.striker_zones = 3
        # Midfield and Striker reward magnitudes
        self.midfield_reward = 0.1
        self.striker_reward = 0.3
        self.midfield_checkpoints_collected = {}
        self.striker_checkpoints_collected = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoints_collected = {}
        self.striker_checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_midfield'] = self.midfield_checkpoints_collected
        to_pickle['CheckpointRewardWrapper_striker'] = self.striker_checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_checkpoints_collected = from_pickle['CheckpointRewardWrapper_midfield']
        self.striker_checkpoints_collected = from_pickle['CheckpointRewardWrapper_striker']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_cooperative_reward": [0.0] * len(reward),
                      "striker_finish_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Adjusting rewards based on cooperative play between midfielders and strikers
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Midfield cooperative strategy:
            if o['active'] in midfielder_indices and o['ball_owned_team'] == 0:  # Assuming 0 is the team with midfield strategy
                midfield_distance = np.sqrt((o['ball'][0] - 0.5) ** 2 + o['ball'][1] ** 2)
                midfield_index = int(midfield_distance * self.midfield_zones)
                
                if midfield_index not in self.midfield_checkpoints_collected:
                    reward[rew_index] += self.midfield_reward
                    self.midfield_checkpoints_collected[midfield_index] = True
                    components["midfield_cooperative_reward"][rew_index] = self.midfield_reward

            # Striker finishing strategy:
            if o['active'] in striker_indices and o['ball_owned_team'] == 0:
                striker_distance = np.sqrt((o['ball'][0] - opposition_goal_x) ** 2 + o['ball'][1] ** 2)
                striker_index = int(striker_distance * self.striker_zones)
                
                if striker_index not in self.striker_checkpoints_collected:
                    reward[rew_index] += self.striker_reward
                    self.striker_checkpoints_collected[striker_index] = True
                    components["striker_finish_reward"][rew_index] = self.striker_reward

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
