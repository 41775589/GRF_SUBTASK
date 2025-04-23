import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized reward for mastering close-range attacks and efficient dribbling against goalkeepers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.goal_position = 1  # Opponent's goal x-coordinate
        self.precision_reward_strength = 0.1
        self.dribble_reward_strength = 0.05
        self.goal_zone_threshold = 0.1  # Threshold distance to goal considered as close-range
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
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
                      "precision_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Close-range precision reward
            if o['ball'][0] > (self.goal_position - self.goal_zone_threshold):
                # Check if the agent is also close to the ball
                if o['active'] == o['ball_owned_player'] and o['ball_owned_team'] == 1:
                    components["precision_reward"][rew_index] += self.precision_reward_strength
                    reward[rew_index] += self.precision_reward_strength
            
            # Dribbling reward
            if o['sticky_actions'][9] == 1:  # Check if dribble action is active
                components["dribble_reward"][rew_index] += self.dribble_reward_strength
                reward[rew_index] += self.dribble_reward_strength
        
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
