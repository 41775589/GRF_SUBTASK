import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for using `Stop-Dribble` to manage ball control under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds or checkpoints
        self.stop_dribble_reward = 0.1
        self.dribble_threshold = 5  # Hypothetical threshold for dribbling under pressure
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Gets observation from the environment
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_by_agent_team = o['ball_owned_team'] == 0

            dribble_action_active = o['sticky_actions'][9] == 1
            non_dribble_action_taken = np.sum(o['sticky_actions']) - o['sticky_actions'][9]

            # Assuming there is some counter or metric representing defensive pressure; if dribbling under pressure
            if controlled_by_agent_team and dribble_action_active and non_dribble_action_taken >= self.dribble_threshold:
                components["stop_dribble_reward"][rew_index] = self.stop_dribble_reward
                reward[rew_index] += components["stop_dribble_reward"][rew_index]
        
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
