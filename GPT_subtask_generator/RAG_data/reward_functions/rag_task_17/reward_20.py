import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that incentivizes wide midfield play and use of high passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self._wide_field_reward = 0.05
        self._pass_success_reward = 0.1
        self._num_zones = 5
        self.collected_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        base_reward = reward.copy()
        components = {"base_score_reward": base_reward, "wide_field_reward": [0.0] * len(reward), "pass_success_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        for idx, agent_obs in enumerate(observation):
            ball_pos = agent_obs['ball'][:2]
            
            # Calculate the field zones for wide play reward (e.g., sidelines)
            if abs(ball_pos[1]) > 0.3:
                zone = round(ball_pos[1] * self._num_zones)
                if zone not in self.collected_zones:
                    self.collected_zones[zone] = True
                    components["wide_field_reward"][idx] += self._wide_field_reward
                    reward[idx] += components["wide_field_reward"][idx]
                    
            # Check for successful high pass (action 2 corresponds to high pass in many envs)
            if agent_obs['sticky_actions'][2] == 1:  # assuming this is the index for high pass action
                components["pass_success_reward"][idx] += self._pass_success_reward
                reward[idx] += components["pass_success_reward"][idx]
                
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
